# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread
import imageio
import os

from . import util

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.blending import BlendParams, sigmoid_alpha_blend, softmax_rgb_blend


def set_rasterizer(type = 'pytorch3d'):
    if type == 'pytorch3d':
        global Meshes, load_obj, rasterize_meshes

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        self.raster_settings_dict = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        self.raster_settings = util.dict2obj(self.raster_settings_dict)

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

        return pixel_vals

    def extra_repr(self):
        return '{image_size}px, blur_radius={blur_radius}, faces_per_pixel={faces_per_pixel}'.format(
            **self.raster_settings_dict)


class FragmentsContainer(object):
    def __init__(self, pix_to_face, zbuf, bary_coords, dists):
        self.pix_to_face = pix_to_face
        self.zbuf = zbuf
        self.bary_coords = bary_coords
        self.dists = dists


class SilhouetteRasterizer(nn.Module):
    def __init__(self, image_size=224, sigma=1e-4):
        super().__init__()
        self.raster_settings_dict = {
            'image_size': image_size,
            'blur_radius': sigma,
            'faces_per_pixel': 10,
        }
        self.raster_settings = util.dict2obj(self.raster_settings_dict)

    def forward(self, vertices, faces):
        batch_size = vertices.shape[0]

        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=self.raster_settings.image_size,
            blur_radius=self.raster_settings.blur_radius,
            faces_per_pixel=self.raster_settings.faces_per_pixel,
        )

        colors = torch.ones_like(bary_coords)
        fragments = FragmentsContainer(pix_to_face, zbuf, bary_coords, dists)

        images = sigmoid_alpha_blend(colors, fragments, BlendParams())

        with torch.no_grad():
            # Find vertices that lie in visibile faces
            dtype = vertices.dtype
            device = vertices.device
            offset = torch.arange(batch_size, device=device)[:, None, None]

            visible_face_idx = pix_to_face[..., 0].view(-1)

            faces_all = faces + vertices.shape[1] * offset
            faces_all = faces_all.view(-1, 3)
            faces_all = torch.cat([faces_all, -torch.ones(1, 3, device=device).long()],
                                  dim=0)  # add dummy face at the end

            visible_vertices_all = faces_all[visible_face_idx].view(-1)

            vertices_visibility = torch.zeros(batch_size * vertices.shape[1] + 1, dtype=dtype, device=device)
            vertices_visibility[visible_vertices_all] = 1.0
            vertices_visibility = vertices_visibility[:-1].view(batch_size, -1)

        return images[..., 3:].permute(0, 3, 1, 2), vertices_visibility

    def extra_repr(self):
        return '{image_size}px, blur_radius={blur_radius}, faces_per_pixel={faces_per_pixel}'.format(
            **self.raster_settings_dict)


class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, files_dir=None, uv_size=256, rasterizer_type='pytorch3d'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.shape_light_intensity = 1.7
        # dark light
        self.light_directions = [
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, -1],
            [1, -1, -1],
            [0, 0, 1]
        ]

        self.light_directions = [
            [-1, 1, 1],
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [0, 0, 1]
        ]

        parent_dir = os.path.dirname(obj_filename) if files_dir is None else files_dir

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        # uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        # faces = faces.verts_idx[None, ...]

        uvfaces = torch.load(f'{parent_dir}/uvfaces_v2.pth')
        faces = torch.load(f'{parent_dir}/faces_v2.pth')

        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(image_size)
        self.silhouette_renderer = SilhouetteRasterizer(image_size)

        # faces
        dense_triangles = util.generate_triangles(self.image_size, self.image_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :], persistent=False)
        self.register_buffer('faces', faces, persistent=False)
        self.register_buffer('raw_uvcoords', uvcoords, persistent=False)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords, persistent=False)
        self.register_buffer('uvfaces', uvfaces, persistent=False)
        self.register_buffer('face_uvcoords', face_uvcoords, persistent=False)

        # Get UV coords for each vertex in the mesh
        self.register_buffer('true_uvcoords', torch.load(f'{parent_dir}/vertex_uvcoords.pth'), persistent=False)

        # shape colors, for rendering shape overlay
        self.predefined_colors = blue_color = [170, 194, 235]
        colors = torch.tensor(self.predefined_colors)[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors, persistent=False)
        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()

        self.register_buffer('constant_factor', constant_factor, persistent=False)

    def forward(self, vertices, transformed_vertices, albedos=None, lights=None, light_type='point', faces=None,
                face_uvcoords=None, face_masks=None, render_only_soft_silhouette=False):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        input_type = vertices.type()

        vertices = vertices.float()
        transformed_vertices = transformed_vertices.float()

        if faces is None:
            faces = self.faces
            face_uvcoords = self.face_uvcoords

        batch_size = vertices.shape[0]

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        (
            soft_alpha_images,
            vertices_visibility
        ) = self.silhouette_renderer(
            transformed_vertices,
            faces.expand(batch_size, -1, -1)
        )

        if not render_only_soft_silhouette:
            # attributes
            face_vertices = util.face_vertices(vertices, faces.expand(batch_size, -1, -1))
            attributes = face_uvcoords.expand(batch_size, -1, -1, -1)

            transformed_normals = util.vertex_normals(transformed_vertices, faces.expand(batch_size, -1, -1))
            transformed_face_normals = util.face_vertices(transformed_normals, faces.expand(batch_size, -1, -1))

            attributes = torch.cat(
                [
                    attributes,
                    transformed_face_normals.detach(),
                    face_vertices.detach()
                ],
                -1
            )

            normals = util.vertex_normals(vertices, faces.expand(batch_size, -1, -1))
            face_normals = util.face_vertices(normals, faces.expand(batch_size, -1, -1))
            attributes = torch.cat([attributes, face_normals], -1)

            if face_masks is not None:
                face_masks = face_masks.expand(batch_size, -1, -1, -1).clone()
                attributes = torch.cat([face_masks, attributes], -1)

            # rasterize
            rendering_ = self.rasterizer(transformed_vertices, faces.expand(batch_size, -1, -1), attributes)

            if face_masks is not None:
                area_alpha_images = (rendering_[:, :face_masks.shape[-1]].detach().clone() > 0.0).float()
                rendering = rendering_[:, face_masks.shape[-1]:]
            else:
                rendering = rendering_

            # vis mask
            alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

            # albedo
            uvcoords_images = rendering[:, :3, :, :]
            normal_images = rendering[:, -4:-1, :, :]
            vertice_images = rendering[:, 6:9, :, :].detach()

            if albedos is not None:
                grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
                albedo_images = F.grid_sample(albedos, grid, align_corners=False)

            if lights is not None:
                # visible mask for pixels with positive normal direction
                transformed_normal_map = rendering[:, 3:6, :, :].detach()
                pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

            if lights is not None:
                # shading
                if lights.shape[1] == 9:
                    shading_images = self.add_SHlight(normal_images, lights)
                else:
                    if light_type == 'point':
                        shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                        shading_images = shading.reshape(
                            [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
                    else:
                        shading = self.add_directionlight(
                            normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                            lights)
                        shading_images = shading.reshape(
                            [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)

            elif albedos is not None:
                images = albedo_images
                shading_images = images.detach() * 0.0

            alpha_images = alpha_images.type(input_type)

            outputs = {
                'alpha_images': alpha_images,
                'area_alpha_images': area_alpha_images.type(input_type) if face_masks is not None else None,
                'soft_alpha_images': soft_alpha_images.type(input_type),
                'uvcoords_images': uvcoords_images.type(input_type) * alpha_images,
                'normal_images': normal_images.type(input_type) * alpha_images,
                'vertice_images': vertice_images.type(input_type) * alpha_images,
                'vertices_visibility': vertices_visibility.type(input_type),
            }

            if albedos is not None:
                outputs['images'] = shading_images * albedo_images
                outputs['albedo_images'] = albedo_images.type(input_type) * alpha_images

            if lights is not None:
                outputs['pos_mask'] = pos_mask.type(input_type)
                outputs['shading_images'] = shading_images.type(input_type)

            return outputs

        else:
            outputs = {
                'area_alpha_images': area_alpha_images.type(input_type) if face_masks is not None else None,
                'soft_alpha_images': soft_alpha_images.type(input_type),
                'vertices_visibility': vertices_visibility.type(input_type)
            }

            return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices,
                     images=None, detail_normal_images=None, lights=None, faces=None, face_uvcoords=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]

        input_type = vertices.type()

        vertices = vertices.float()
        transformed_vertices = transformed_vertices.float()

        if faces is None:
            faces = self.faces
            face_uvcoords = self.face_uvcoords
            face_colors = self.face_colors

        else:
            colors = torch.tensor(self.predefined_colors)[None, None, :].repeat(1, faces.max() + 1, 1).float().to(
                faces.device) / 255.
            face_colors = util.face_vertices(colors, faces)

        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                self.light_directions
            )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float() * self.shape_light_intensity
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, faces.expand(batch_size, -1, -1))
        attributes = torch.cat([face_colors.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertices.detach(),
                                face_normals],
                               -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).contiguous()
        shaded_images = albedo_images * shading_images

        alpha_images = alpha_images * pos_mask
        if images is None:
            shape_images = shaded_images * alpha_images + torch.ones_like(shaded_images).to(vertices.device) * (
                    1 - alpha_images)
        else:
            shape_images = shaded_images * alpha_images + images * (1 - alpha_images)

        return shape_images.type(input_type)

    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3).clone()
        z = z - z.min()
        z = z / z.max()
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_colors(self, transformed_vertices, colors):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = colors.shape[0]

        # Attributes
        attributes = util.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :] * alpha_images
        return images

    def world2uv(self, vertices, faces=None, uvcoords=None, uvfaces=None):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        if uvcoords is None:
            faces = self.faces
            uvcoords = self.uvcoords
            uvfaces = self.uvfaces

        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, faces.expand(batch_size, -1, -1))

        uv_vertices = self.uv_rasterizer(uvcoords.expand(batch_size, -1, -1),
                                         uvfaces.expand(batch_size, -1, -1),
                                         face_vertices)[:, :3]
        return uv_vertices
#include "redner.h"
#include "active_pixels.h"
#include "area_light.h"
#include "automatic_uv_map.h"
#include "camera.h"
#include "camera_distortion.h"
#include "envmap.h"
#include "vmf.h"
#include "load_serialized.h"
#include "material.h"
#include "pathtracer.h"
#include "pathtracer_was.h"
#include "ptr.h"
#include "scene.h"
#include "shape.h"
#include "quadric.h"
#include "rejection_test.h"
#include "lp_solve.h"
#include "offset_quadric.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(redner, m) {
    m.doc() = "Redner"; // optional module docstring

    py::class_<ptr<float>>(m, "float_ptr", py::module_local())
        .def(py::init<std::size_t>())
        .def("allocate", &ptr<float>::allocate)
        .def("destroy_array", &ptr<float>::destroy_array)
        .def("set_index", &ptr<float>::set_index)
        .def("get_index", &ptr<float>::get_index)
        .def("get_pointer", &ptr<float>::get_pointer)
        .def("copy", &ptr<float>::copy);

    py::class_<ptr<double>>(m, "double_ptr", py::module_local())
        .def(py::init<std::size_t>())
        .def("allocate", &ptr<double>::allocate)
        .def("destroy_array", &ptr<double>::destroy_array)
        .def("set_index", &ptr<double>::set_index)
        .def("get_index", &ptr<double>::get_index)
        .def("get_pointer", &ptr<double>::get_pointer);

    py::class_<ptr<int>>(m, "int_ptr", py::module_local())
        .def(py::init<std::size_t>())
        .def("allocate", &ptr<int>::allocate)
        .def("destroy_array", &ptr<int>::destroy_array)
        .def("set_index", &ptr<int>::set_index)
        .def("get_index", &ptr<int>::get_index)
        .def("get_pointer", &ptr<int>::get_pointer);

    py::class_<ptr<Vector3>>(m, "Vector3_ptr", py::module_local())
        .def(py::init<std::size_t>())
        .def("allocate", &ptr<Vector3>::allocate)
        .def("destroy_array", &ptr<Vector3>::destroy_array)
        .def("set_index", &ptr<Vector3>::set_index)
        .def("get_index", &ptr<Vector3>::get_index)
        .def("get_pointer", &ptr<Vector3>::get_pointer);

    py::class_<ptr<Vector4>>(m, "Vector4_ptr", py::module_local())
        .def(py::init<std::size_t>())
        .def("allocate", &ptr<Vector4>::allocate)
        .def("destroy_array", &ptr<Vector4>::destroy_array)
        .def("set_index", &ptr<Vector4>::set_index)
        .def("get_index", &ptr<Vector4>::get_index)
        .def("get_pointer", &ptr<Vector4>::get_pointer);

    py::enum_<CameraType>(m, "CameraType", py::module_local())
        .value("perspective", CameraType::Perspective)
        .value("orthographic", CameraType::Orthographic)
        .value("fisheye", CameraType::Fisheye)
        .value("panorama", CameraType::Panorama);

    py::enum_<FilterType>(m, "FilterType", py::module_local())
        .value("box", FilterType::Box)
        .value("gaussian", FilterType::Gaussian);

    py::class_<Camera>(m, "Camera", py::module_local())
        .def(py::init<int,
                      int,
                      ptr<float>, // position
                      ptr<float>, // look
                      ptr<float>, // up
                      ptr<float>, // cam_to_world
                      ptr<float>, // world_to_cam
                      ptr<float>, // ndc_to_cam
                      ptr<float>, // cam_to_ndc
                      ptr<float>, // distortion_params
                      float, // clip_near
                      CameraType,
                      FilterType,
                      Vector2i, // viewport_beg
                      Vector2i>()) // viewport_end
        .def_readonly("use_look_at", &Camera::use_look_at)
        .def("has_distortion_params", &Camera::has_distortion_params);

    py::class_<DCamera>(m, "DCamera", py::module_local())
        .def(py::init<ptr<float>, // position
                      ptr<float>, // look
                      ptr<float>, // up
                      ptr<float>, // cam_to_world
                      ptr<float>, // world_to_cam
                      ptr<float>, // ndc_to_cam
                      ptr<float>, // cam_to_ndc
                      ptr<float>>()); // distortion_params

    py::class_<Scene>(m, "Scene", py::module_local())
        .def(py::init<const Camera &,
                      const std::vector<const Shape*> &,
                      const std::vector<const Material*> &,
                      const std::vector<const AreaLight*> &,
                      const std::shared_ptr<const EnvironmentMap> &,
                      const std::shared_ptr<const VonMisesFisherLight> &,
                      bool,
                      int>())
        .def_readonly("max_generic_texture_dimension",
            &Scene::max_generic_texture_dimension)
        .def_readwrite("remove_concave", &Scene::remove_concave);

    py::class_<DScene, std::shared_ptr<DScene>>(m, "DScene", py::module_local())
        .def(py::init<const DCamera &,
                      const std::vector<DShape*> &,
                      const std::vector<DMaterial*> &,
                      const std::vector<DAreaLight*> &,
                      const std::shared_ptr<DEnvironmentMap> &,
                      const std::shared_ptr<DVonMisesFisherLight> &,
                      bool,
                      int>());

    py::class_<Shape>(m, "Shape", py::module_local())
        .def(py::init<ptr<float>, // vertices
                      ptr<int>, // indices
                      ptr<float>, // uvs
                      ptr<float>, // normals
                      ptr<int>, // uv_indices
                      ptr<int>, // normal_indices
                      ptr<float>, // colors
                      int, // num_vertices
                      int, // num_uv_vertices
                      int, // num_normal_vertices
                      int, // num_triangles
                      int, // material_id
                      int  // light_id
                      >())
        .def_readonly("num_vertices", &Shape::num_vertices)
        .def_readonly("num_uv_vertices", &Shape::num_uv_vertices)
        .def_readonly("num_normal_vertices", &Shape::num_normal_vertices)
        .def("has_uvs", &Shape::has_uvs)
        .def("has_normals", &Shape::has_normals)
        .def("has_colors", &Shape::has_colors);

    py::class_<DShape>(m, "DShape", py::module_local())
        .def(py::init<ptr<float>,
                      ptr<float>,
                      ptr<float>,
                      ptr<float>>());

    py::class_<Texture1>(m, "Texture1", py::module_local())
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<Texture3>(m, "Texture3", py::module_local())
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<TextureN>(m, "TextureN", py::module_local())
        .def(py::init<const std::vector<ptr<float>> &,
                      const std::vector<int> &, // width
                      const std::vector<int> &, // height
                      int, // channels
                      ptr<float>>());

    py::class_<Material>(m, "Material", py::module_local())
        .def(py::init<Texture3, // diffuse
                      Texture3, // specular
                      Texture1, // roughness
                      TextureN, // generic_texture
                      Texture3, // normal_map
                      bool, // compute_specular_lighting
                      bool, // two_sided
                      bool>()) // use_vertex_color
        .def("get_diffuse_levels", &Material::get_diffuse_levels)
        .def("get_diffuse_size", &Material::get_diffuse_size)
        .def("get_specular_levels", &Material::get_specular_levels)
        .def("get_specular_size", &Material::get_specular_size)
        .def("get_roughness_levels", &Material::get_roughness_levels)
        .def("get_roughness_size", &Material::get_roughness_size)
        .def("get_generic_levels", &Material::get_generic_levels)
        .def("get_generic_size", &Material::get_generic_size)
        .def("get_normal_map_levels", &Material::get_normal_map_levels)
        .def("get_normal_map_size", &Material::get_normal_map_size);

    py::class_<DMaterial>(m, "DMaterial", py::module_local())
        .def(py::init<Texture3, // diffuse
                      Texture3, // specular
                      Texture1, // roughness
                      TextureN, // generic_texture
                      Texture3>()); // normal_map

    py::class_<AreaLight>(m, "AreaLight", py::module_local())
        .def(py::init<int, // shape_id
                      ptr<float>, // intensity
                      bool, // two_sided
                      bool>()); // directly_visible

    py::class_<DAreaLight>(m, "DAreaLight", py::module_local())
        .def(py::init<ptr<float>>());

    py::class_<EnvironmentMap, std::shared_ptr<EnvironmentMap>>(m, "EnvironmentMap", py::module_local())
        .def(py::init<Texture3,   // values
                      ptr<float>, // env_to_world
                      ptr<float>, // world_to_env
                      ptr<float>, // sample_cdf_ys
                      ptr<float>, // sample_cdf_xs
                      Real, // pdf_norm
                      bool>()) // directly_visible
        .def("get_levels", &EnvironmentMap::get_levels)
        .def("get_size", &EnvironmentMap::get_size);
    py::class_<DEnvironmentMap, std::shared_ptr<DEnvironmentMap>>(m, "DEnvironmentMap", py::module_local())
        .def(py::init<Texture3,       // values
                      ptr<float>>()); // world_to_env

    py::class_<VonMisesFisherLight, std::shared_ptr<VonMisesFisherLight>>(m, "VonMisesFisherLight", py::module_local())
        .def(py::init<Real,   // kappa
                      ptr<float>, // intensity_data
                      ptr<float>, // env_to_world
                      ptr<float>, // world_to_env
                      Real>());
    py::class_<DVonMisesFisherLight, std::shared_ptr<DVonMisesFisherLight>>(m, "DVonMisesFisherLight", py::module_local())
        .def(py::init<ptr<float>,       // kappa
                      ptr<float>,       // intensity
                      ptr<float>>()); // world_to_env

    py::enum_<Channels>(m, "channels", py::module_local())
        .value("radiance", Channels::radiance)
        .value("alpha", Channels::alpha)
        .value("depth", Channels::depth)
        .value("position", Channels::position)
        .value("geometry_normal", Channels::geometry_normal)
        .value("shading_normal", Channels::shading_normal)
        .value("uv", Channels::uv)
        .value("barycentric_coordinates", Channels::barycentric_coordinates)
        .value("diffuse_reflectance", Channels::diffuse_reflectance)
        .value("specular_reflectance", Channels::specular_reflectance)
        .value("roughness", Channels::roughness)
        .value("generic_texture", Channels::generic_texture)
        .value("vertex_color", Channels::vertex_color)
        .value("shape_id", Channels::shape_id)
        .value("triangle_id", Channels::triangle_id)
        .value("material_id", Channels::material_id);

    m.def("compute_num_channels", compute_num_channels, "");

    py::enum_<SamplerType>(m, "SamplerType", py::module_local())
        .value("independent", SamplerType::independent)
        .value("sobol", SamplerType::sobol);

    py::class_<edge_sampling::RenderOptions>(m, "RenderOptions", py::module_local())
        .def(py::init<uint64_t,
                      int, // num_samples
                      int, // max_bounces
                      std::vector<Channels>,
                      SamplerType,
                      bool, // sample_pixel_center
                      bool, // use_primary_edge_sampling
                      bool, // use_secondary_edge_sampling
                      bool // remove_concave
                      >())
        .def_readwrite("seed", &edge_sampling::RenderOptions::seed)
        .def_readwrite("num_samples", &edge_sampling::RenderOptions::num_samples);

    py::enum_<vfield::ImportanceSampling>(m, "ImportanceSamplingVField", py::module_local())
        .value("cosine_hemisphere", vfield::ImportanceSampling::cosine_hemisphere);

    py::enum_<vfield::VarianceReduction>(m, "VarianceReductionVField", py::module_local())
        .value("none", vfield::VarianceReduction::none)
        .value("antithetic_variate", vfield::VarianceReduction::antithetic_variate);

    py::class_<vfield::VarianceReductionSettings>(m, "VarianceReductionSettings", py::module_local())
        .def(py::init<bool,
                      bool,
                      bool,
                      bool,
                      bool,
                      int>());

    py::class_<vfield::RenderOptions>(m, "RenderOptionsVField", py::module_local())
        .def(py::init<uint64_t,
                      int,
                      int,
                      std::vector<Channels>,
                      SamplerType,
                      SamplerType,
                      vfield::VarianceReductionSettings,
                      vfield::ImportanceSampling,
                      KernelParameters,
                      bool,
                      bool,
                      bool>())
        .def_readwrite("seed", &vfield::RenderOptions::seed)
        .def_readwrite("num_samples", &vfield::RenderOptions::num_samples);

    py::class_<KernelParameters>(m, "KernelParameters", py::module_local())
        .def(py::init<Real,
                      Real,
                      Real,
                      Real,
                      Real,
                      int,
                      Real,
                      int,
                      bool,
                      Real,
                      int,
                      bool,
                      bool,
                      Real>());

        
    py::class_<Vector2i>(m, "Vector2i", py::module_local())
        .def(py::init<int, int>())
        .def_readwrite("x", &Vector2i::x)
        .def_readwrite("y", &Vector2i::y);

    py::class_<Vector2f>(m, "Vector2f", py::module_local())
        .def(py::init<float, float>())
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y);

    py::class_<Vector3f>(m, "Vector3f", py::module_local())
        .def(py::init<float, float, float>())
        .def(py::self + py::self)
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z);

    py::class_<Vector4f>(m, "Vector4f", py::module_local())
        .def(py::init<float, float, float, float>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def_readwrite("x", &Vector4f::x)
        .def_readwrite("y", &Vector4f::y)
        .def_readwrite("z", &Vector4f::z)
        .def_readwrite("w", &Vector4f::w);

    py::class_<Vector3>(m, "Vector3", py::module_local())
        .def(py::init<double, double, double>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z);

    py::class_<Vector4>(m, "Vector4", py::module_local())
        .def(py::init<double, double, double, double>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def_readwrite("x", &Vector4::x)
        .def_readwrite("y", &Vector4::y)
        .def_readwrite("z", &Vector4::z)
        .def_readwrite("w", &Vector4::w);

    py::class_<Matrix3x3f>(m, "Matrix3x3f", py::module_local())
        .def(py::init<float, float, float,
                      float, float, float,
                      float, float, float>())
        .def("col", &Matrix3x3f::col)
        .def("row", &Matrix3x3f::row)
        .def("__call__", [](const Matrix3x3f &m, int i, int j){
                   return m(i, j);
                 }
            );

    py::class_<Matrix3x3>(m, "Matrix3x3", py::module_local())
        .def(py::init<double, double, double,
                      double, double, double,
                      double, double, double>())
        .def("col", &Matrix3x3::col)
        .def("row", &Matrix3x3::row)
        .def("__call__", [](const Matrix3x3 &m, int i, int j){
                   return m(i, j);
                 }
            );

    py::class_<Matrix4x4f>(m, "Matrix4x4f", py::module_local())
        .def(py::init<float, float, float, float,
                      float, float, float, float,
                      float, float, float, float,
                      float, float, float, float>())
        
        .def("__call__", [](const Matrix4x4f &m, int i, int j){
                   return m(i, j);
                 }
            );
        

    py::class_<Matrix4x4>(m, "Matrix4x4", py::module_local())
        .def(py::init<double, double, double, double,
                      double, double, double, double,
                      double, double, double, double,
                      double, double, double, double>())
        
        .def("__call__", [](const Matrix4x4 &m, int i, int j){
                   return m(i, j);
                 }
            )
        .def("identity", &Matrix4x4::identity);
    py::class_<MitsubaTriMesh>(m, "MitsubaTriMesh", py::module_local())
        .def_readwrite("vertices", &MitsubaTriMesh::vertices)
        .def_readwrite("indices", &MitsubaTriMesh::indices)
        .def_readwrite("uvs", &MitsubaTriMesh::uvs)
        .def_readwrite("normals", &MitsubaTriMesh::normals);

    m.def("load_serialized", &load_serialized, "");

    // For auto uv unwrapping
    py::class_<UVTriMesh>(m, "UVTriMesh", py::module_local())
        .def(py::init<ptr<float>, // vertices
                      ptr<int>, // indices
                      ptr<float>, // uvs
                      ptr<int>, // uv_indices
                      int, // num_vertices
                      int, // num_uv_vertices
                      int>()) // num_triangles
        .def_readwrite("uvs", &UVTriMesh::uvs)
        .def_readwrite("uv_indices", &UVTriMesh::uv_indices)
        .def_readwrite("num_uv_vertices", &UVTriMesh::num_uv_vertices);
    py::class_<TextureAtlas>(m, "TextureAtlas", py::module_local())
        .def(py::init<>());
    m.def("automatic_uv_map", &automatic_uv_map, "");
    m.def("copy_texture_atlas", &copy_texture_atlas, "");

    m.def("render", &edge_sampling::render, "");
    m.def("render_warped", &vfield::render, "");

    py::class_<Conic>(m, "Conic", py::module_local())
        .def(py::init<Matrix3x3&>())
        .def_readwrite("matrix", &Conic::matrix);

    py::class_<Quadric>(m, "Quadric", py::module_local())
        .def(py::init<Matrix4x4&>())
        .def_readwrite("matrix", &Quadric::matrix);

    py::class_<QuadricPair>(m, "QuadricPair", py::module_local())
        .def(py::init<Quadric&, Quadric&>())
        .def_readwrite("quadric1", &QuadricPair::quadric1)
        .def_readwrite("quadric2", &QuadricPair::quadric2);

    py::class_<ConicIntersection>(m, "ConicIntersection", py::module_local())
        .def(py::init<Vector3, Vector3>())
        .def_readwrite("p1", &ConicIntersection::p1)
        .def_readwrite("p2", &ConicIntersection::p2);

    py::class_<QuadricIntersection>(m, "QuadricIntersection", py::module_local())
        .def(py::init<Vector4, Vector4>())
        .def_readwrite("p1", &QuadricIntersection::p1)
        .def_readwrite("p2", &QuadricIntersection::p2);

    py::class_<AABB3>(m, "AABB3", py::module_local())
        .def(py::init<Vector3, Vector3>())
        .def_readwrite("p_min", &AABB3::p_min)
        .def_readwrite("p_max", &AABB3::p_max);
    m.def("corner", &corner, "");

    py::class_<MinSphere3D>(m, "MinSphere3D", py::module_local())
        .def_readwrite("center", &MinSphere3D::center)
        .def_readwrite("radius", &MinSphere3D::radius);

    m.def("intersect_with_line", &intersect_with_line<double>, "");
    m.def("get_point_on_conic", &get_point_on_conic<double>, "");
    m.def("get_conic_from_quadric", &get_conic_from_quadric<double>, "");
    m.def("intersect_with_segment", &intersect_with_segment<double>, "");
    m.def("solve_gen_eig_py", &solve_gen_eig_py<double>, "");
    m.def("fit_quadric_file_input", &fit_quadric_file_input<double>, "");
    m.def("compute_LU_py", &compute_LU_py<double>, "");
    m.def("solve_LU_py", &solve_LU_py<double>, "");
    m.def("find_basis", static_cast<TMatrix3x3<double> (*)(const TVector3<double>&)>(&find_basis<double>), "");
    m.def("test_aabb", &test_aabb, "");
    m.def("rejection_test_py", &rejection_test_py<double>, "");
    m.def("test_solve_lp", &test_solve_lp<double>, "");
    m.def("test_find_offset_quadric_vector", &test_find_offset_quadric_vector<double>, "");
    m.def("test_find_approx_bounding_sphere", &test_find_approx_bounding_sphere<double>, "");
    /// Tests
    m.def("test_sample_primary_rays", &test_sample_primary_rays, "");
    m.def("test_scene_intersect", &test_scene_intersect, "");
    m.def("test_sample_point_on_light", &test_sample_point_on_light, "");
    m.def("test_active_pixels", &test_active_pixels, "");
    m.def("test_camera_derivatives", &test_camera_derivatives, "");
    m.def("test_camera_distortion", &test_camera_distortion, "");
    m.def("test_d_bsdf", &test_d_bsdf, "");
    m.def("test_d_bsdf_sample", &test_d_bsdf_sample, "");
    m.def("test_d_bsdf_pdf", &test_d_bsdf_pdf, "");
    m.def("test_d_intersect", &test_d_intersect, "");
    m.def("test_d_sample_shape", &test_d_sample_shape, "");
    m.def("test_atomic", &test_atomic, "");
}

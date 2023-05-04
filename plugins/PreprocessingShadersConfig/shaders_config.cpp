#include "shaders_config.h"

using namespace std;

//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

string g_current_filepath;

afCameraHMD::afCameraHMD()
{
    // For HTC Vive Pro
    m_width = 2880;
    m_height = 1600;
    m_alias_scaling = 1.0;
}

int afCameraHMD::init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs)
{
    m_camera = (afCameraPtr)a_afObjectPtr;
    m_world_ptr = m_camera->m_afWorld;
    afRigidBodyVec rigid_bodies_vec = m_world_ptr->getRigidBodies();

    cout << "\n\n\n\n\n";
    cout << "Shaders config plugin initialized" << endl;
    cout << "camera name:   " << m_camera->getName() << endl;
    cout << "camera parent: " << m_camera->m_parentName << endl;
    cout << "World name:    " << m_world_ptr->getName() << endl;
    cout << "List of objects in simulation" << endl;
    cout << "Number of objects: " << rigid_bodies_vec.size() << endl;
    cout << "\n\n\n\n\n";

    return 1;
}

void afCameraHMD::graphicsUpdate()
{
    static bool first_time = true;
    if (first_time)
    {
        cout << "first time in the graphics update" << endl;
        cout << "World name:    " << m_world_ptr->getName() << endl;

        afBaseObjectMap *rigid_bodies_map = m_world_ptr->getRigidBodyMap();
        cout << "Number of objects in simulation:"
             << rigid_bodies_map->size() << endl;

        afBaseObjectMap::iterator it = rigid_bodies_map->begin();
        // std::pair<std::string, afBaseObjectPtr> pair;

        for (std::pair<std::string, afBaseObjectPtr> pair : *rigid_bodies_map)
        {
            cout << "Object name: " << pair.first << endl;
        }

        string needle_name = "/ambf/env/BODY Needle";
        afRigidBodyPtr needle_ptr = dynamic_cast<afRigidBody *>(rigid_bodies_map->at(needle_name));
        afRigidBodyAttributes *needle_attr = dynamic_cast<afRigidBodyAttributes *>(needle_ptr->getAttributes());

        // m_visualAttribs m_colorAttribs  m_diffuse
        needle_diffuse = needle_attr->m_visualAttribs.m_colorAttribs.m_diffuse;
        double r, g, b;
        needle_diffuse.getXYZ(r, g, b);
        needle_attr->m_visualAttribs.m_colorAttribs.m_diffuse.set(0.0, g, 0.0);

        cout << "Needle diffuse color: " << endl;
        needle_diffuse.print();

        //  << m_diffuse[0] << ", "
        //  << m_diffuse[1] << ", "
        //  << m_diffuse[2] << ", " << endl;

        // afRigidBodyVec rigid_bodies_vec = m_world_ptr->getRigidBodies();
        // cout << "Number of objects in simulation:"
        //      << m_world_ptr->getRigidBodies().size() << endl;

        // afRigidBodyVec::iterator it = rigid_bodies_vec.begin();
        // for (it; it != rigid_bodies_vec.end(); it++)
        // {
        //     cout << "Object name: " << (*it)->getName() << endl;

        //     if ((*it)->getName() == "Needle")
        //     {
        //         cout << "Found the needle" << endl;
        //     }
        // }
        // cout << "First object" << rigid_bodies_vec[0]->getName() << endl;

        first_time = false;
    }
}

void afCameraHMD::physicsUpdate(double dt)
{
}

void afCameraHMD::reset()
{
}

bool afCameraHMD::close()
{
    return true;
}

// void afCameraHMD::updateHMDParams()
// {
//     GLint id = m_shaderPgm->getId();
//     //    cerr << "INFO! Shader ID " << id << endl;
//     glUseProgram(id);
//     glUniform1i(glGetUniformLocation(id, "warpTexture"), 2);
//     glUniform2fv(glGetUniformLocation(id, "ViewportScale"), 1, m_viewport_scale);
//     glUniform3fv(glGetUniformLocation(id, "aberr"), 1, m_aberr_scale);
//     glUniform1f(glGetUniformLocation(id, "WarpScale"), m_warp_scale * m_warp_adj);
//     glUniform4fv(glGetUniformLocation(id, "HmdWarpParam"), 1, m_distortion_coeffs);
//     glUniform2fv(glGetUniformLocation(id, "LensCenterLeft"), 1, m_left_lens_center);
//     glUniform2fv(glGetUniformLocation(id, "LensCenterRight"), 1, m_right_lens_center);
// }

// void afCameraHMD::makeFullScreen()
// {
//     const GLFWvidmode *mode = glfwGetVideoMode(m_camera->m_monitor);
//     int w = 2880;
//     int h = 1600;
//     int x = mode->width - w;
//     int y = mode->height - h;
//     int xpos, ypos;
//     glfwGetMonitorPos(m_camera->m_monitor, &xpos, &ypos);
//     x += xpos;
//     y += ypos;
//     glfwSetWindowPos(m_camera->m_window, x, y);
//     glfwSetWindowSize(m_camera->m_window, w, h);
//     m_camera->m_width = w;
//     m_camera->m_height = h;
//     glfwSwapInterval(0);
//     cerr << "\t Making " << m_camera->getName() << " fullscreen \n";
// }

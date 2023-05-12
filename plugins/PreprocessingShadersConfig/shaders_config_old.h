#define GL_SILENCE_DEPRECATION
#include <afFramework.h>
#include <afAttributes.h>

using namespace std;
using namespace ambf;

class afCameraHMD : public afObjectPlugin
{
public:
    afCameraHMD();
    virtual int init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs) override;
    virtual void graphicsUpdate() override;
    virtual void physicsUpdate(double dt) override;
    virtual void reset() override;
    virtual bool close() override;

    void updateHMDParams();

    void makeFullScreen();

protected:
    afCameraPtr m_camera;
    afWorldPtr m_world_ptr;
    afVector3d needle_diffuse;

    cFrameBufferPtr m_frameBuffer;
    cWorld *m_vrWorld;
    cMesh *m_quadMesh;
    int m_width;
    int m_height;
    int m_alias_scaling;
    cShaderProgramPtr m_shaderPgm;

protected:
    float m_viewport_scale[2];
    float m_distortion_coeffs[4];
    float m_aberr_scale[3];
    float m_sep;
    float m_left_lens_center[2];
    float m_right_lens_center[2];
    float m_warp_scale;
    float m_warp_adj;
    float m_vpos;
};

AF_REGISTER_OBJECT_PLUGIN(afCameraHMD)
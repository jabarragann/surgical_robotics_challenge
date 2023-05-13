#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

using namespace std;

int main()
{
    string path = "./../ADF/world/world_stereo.yaml";
    YAML::Node config = YAML::LoadFile(path);

    cout << boolalpha << "config['CameraL2'].IsDefined: " << config["CameraL2"].IsDefined() << endl;
    cout << boolalpha << "config.IsMap(): " << config.IsMap() << endl;

    YAML::Node camera_config = config["cameraL2"];

    if (camera_config.IsMap())
    {
        YAML::Node::const_iterator it = camera_config.begin();
        for (it; it != camera_config.end(); ++it)
        {
            cout << "key: " << it->first.as<string>() << endl;
            // cout << "value: " << it->second.as<string>() << endl;
        }
    }

    YAML::Node shaders_config = camera_config["preprocessing shaders config"];
    if (shaders_config.IsMap())
    {
        cout << shaders_config.size() << endl;
        YAML::Node::const_iterator it = shaders_config.begin();

        for (it; it != shaders_config.end(); it++)
        {
            string element = it->first.as<string>();
            cout << "key: " << element << endl;

            vector<int> rgb_vals = shaders_config[element]["rgb"].as<vector<int>>();
            if (shaders_config[element]["namespace"].IsDefined())
            {
                string af_namespace = shaders_config[element]["namespace"].as<string>();
                cout << "namespace: " << af_namespace << endl;
            }

            // printing
            for (std::size_t i = 0; i < rgb_vals.size(); ++i)
            {
                std::cout << rgb_vals[i] << " ";
            }
            std::cout << std::endl;
        }
        cout << "Map!" << endl;
    }

    // YAML::Node shaders_config = camera_config["preprocessing_shaders_config"];
    // if (shaders_config.IsSequence())
    // {
    //     cout << shaders_config.size() << endl;
    //     for (int i = 0; i < shaders_config.size(); i++)
    //     {
    //         YAML::Node::const_iterator it = shaders_config[i].begin();
    //         for (it; it != shaders_config[i].end(); ++it)
    //         {
    //             string element = it->first.as<string>();
    //             cout << "key: " << element << endl;

    //             vector<int> rgb_vals = shaders_config[i][element]["rgb"].as<vector<int>>();
    //             string af_namespace = shaders_config[i][element]["namespace"].as<string>();
    //             cout << "namespace: " << af_namespace << endl;
    //             for (std::size_t i = 0; i < rgb_vals.size(); ++i)
    //             {
    //                 std::cout << rgb_vals[i] << " ";
    //             }
    //             std::cout << std::endl;
    //             // cout << "value: " << element.second << endl;
    //         }
    //         //  shaders_config[i]
    //     }
    //     cout << "Sequence!" << endl;

    // YAML::Node::const_iterator it = camera_config.begin();
    // for (it; it != camera_config.end(); ++it)
    // {
    //     cout << "key: " << it->first.as<string>() << endl;
    //     // cout << "value: " << it->second.as<string>() << endl;
    // }

    // cout << boolalpha << "config.IsSequence(): " << config.IsMap() << endl;

    // if (config["CameraL2"].IsDefined())
    // {
    //     cout << "config file found" << endl;
    // }
    // else
    // {
    //     cout << "not found" << endl;
    // }

    cout << "finishes" << endl;

    return 0;
}
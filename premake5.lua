workspace "MaxOCR"
    architecture "x64"
    
    configurations
    {
        "Debug",
        "Release"
    }

project "MaxOCR"
    location "MaxOCR"
    kind "ConsoleApp"
    language "C++"
    
    targetdir ("bin/%{cfg.buildcfg}/%{prj.name}")
    objdir ("obj/%{cfg.buildcfg}/%{prj.name}")

    files
    {
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.h"
    }

    includedirs
    {
        -- "vendor/opencv/build/include"
    }

    libdirs
    {
        -- "vendor/opencv/build/x64/vc15/lib"
    }

    links
    {
	-- "opencv_world430.lib",
	-- "opencv_world430d.lib"
    }

    filter "configurations:Debug"
        symbols "On"

    filter "configurations:Release"
        optimize "On"
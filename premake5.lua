-- premake5.lua

-- Ask the active python interpreter where its headers/libs live
local py_include = os.outputof("python -c \"import sysconfig; print(sysconfig.get_paths()['include'])\"")
local py_libdir  = os.outputof("python -c \"import sysconfig; print(sysconfig.get_config_var('installed_base') + '\\\\libs')\"")

workspace "synaption"
    configurations { "Debug", "Release" }
    platforms { "x64" }
    architecture "x86_64"
    location "build"
    cppdialect "C++20"
    staticruntime "Off"

project "synaption_core"
    kind "SharedLib"
    language "C++"
    location "build/synaption_core"
    targetdir "build/bin/%{cfg.buildcfg}"
    targetextension ".pyd"        -- required so Python can import it
    targetprefix ""               -- no "lib" prefix on Windows anyway, but be explicit

    files {
        "src/core/**.h",
        "src/core/**.cpp",
        "src/bindings/**.cpp"
    }

    includedirs {
        "src/core",
        "extern/pybind11/include",
        py_include
    }

    libdirs { py_libdir }

    filter "system:windows"
        systemversion "latest"
        -- Python debug builds need pythonXY_d.lib; simplest is to always
        -- link release python libs even in Debug config (standard pybind11 approach)
        defines { "Py_LIMITED_API=0" }

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
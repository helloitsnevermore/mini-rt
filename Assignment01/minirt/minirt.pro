QT -= core gui

TARGET = minirt
TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++14

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES +=  \
    camera.cpp \
    color.cpp \
    image.cpp \
    jpge.cpp \
    material.cpp \
    point_3d.cpp \
    point_light.cpp \
    ray.cpp \
    sampler.cpp \
    scene.cpp \
    scene_loader.cpp \
    sphere.cpp \
    vector_3d.cpp \
    view_plane.cpp

HEADERS += didal.h  \
    camera.h \
    color.h \
    image.h \
    jpge.h \
    material.h \
    minirt.h \
    point_3d.h \
    point_light.h \
    ray.h \
    sampler.h \
    scene.h \
    scene_loader.h \
    sphere.h \
    vector_3d.h \
    view_plane.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

[app]

# (str) Title of your application
title = W@Home Hive

# (str) Package name
package.name = wathome

# (str) Package domain (needed for android/ios packaging)
package.domain = com.akataleptos

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,json

# (str) Application versioning
version = 1.0.1

# (list) Application requirements
# scipy removed — requires legacy NDK r21e with gfortran for LAPACK cross-compilation.
# requests removed — p4a's cross-compiled python can't pip-install (SSL unavailable).
# Using pure-Python Lanczos eigensolver fallback with numpy, urllib.request for HTTP.
# pyjnius needed for foreground service notification
requirements = python3,kivy,numpy,pillow,pyjnius

# (str) Presplash of the application (used during loading)
#presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
icon.filename = %(source.dir)s/icon-512.png

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

#
# Android specific
#

# (list) Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE,WAKE_LOCK,FOREGROUND_SERVICE,POST_NOTIFICATIONS,FOREGROUND_SERVICE_DATA_SYNC

# (list) Background services to start
services = Worker:service/main.py:foreground

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK / AAB will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (int) Android SDK version to use (buildozer auto-downloads)
#android.sdk = 31

# (str) python-for-android branch to use
#p4a.branch = develop

# (str) Bootstrap to use for android
#p4a.bootstrap = sdl2

# (list) android.archs - list of architectures to build for
# Build arm64-v8a only (modern phones, simpler build)
android.archs = arm64-v8a

# (bool) Accept SDK license automatically
android.accept_sdk_license = True

# (str) Android logcat filters to use
android.logcat_filters = *:S python:D

# (bool) If True, then skip trying to update the Android sdk
# This can be useful to avoid excess Internet downloads or save time
# when an update is due and you just want to test/build your package
# android.skip_update = False

# (bool) If True, then automatically accept SDK license
# agreements. This is intended for automation only. If set to False,
# the default, you will be shown the license when first running
# buildozer.
# android.accept_sdk_license = False

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
# In past, was `android.arch` as we weren't supporting builds for multiple archs at the same time.
# android.arch = arm64-v8a

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.kivy.android.PythonActivity

# (str) Full name including package path of the Java class that implements Android Activity
# use that parameter together with android.entrypoint to set custom Java class instead of PythonActivity
#android.activity_class_name = org.kivy.android.PythonActivity

# (list) Pattern to whitelist for the whole project
#android.whitelist =

# (list) List of Java .jar files to add to the libs so that pyjnius can access
# their classes. Don't add jars that you do not need, since extra jars can slow
# down the build process. Allows wildcards matching, for example:
# OUYA-ODK/libs/*.jar
#android.add_jars = foo.jar,bar.jar,path/to/more/*.jar

# (list) List of Java files to add to the android project (can be java or a
# temporary directory which contains the files to add)
#android.add_src =

# (list) Gradle dependencies to add
#android.gradle_dependencies =

# (bool) Enable AndroidX support. Enable when 'android.gradle_dependencies'
# contains an pointandroid library (e.g. the default 'androidx' ones)
# or any other library that requires it.
# android.enable_androidx requires android.api >= 28
android.enable_androidx = True

# (str) The format used to package the app for release mode (aab or apk or aar).
android.release_artifact = apk

# (str) The format used to package the app for debug mode (apk or aar).
android.debug_artifact = apk

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage, absolute or relative to spec file
# build_dir = ./.buildozer

# (str) Path to build output (i.e. .apk, .aab, .ipa) storage
# bin_dir = ./bin

#!/bin/bash
# Build W@Home Hive APK
# Requirements: JDK 17.0.13+, buildozer, Cython 0.29.x

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use JDK 17.0.13 (17.0.2 has JvmWideVariable bug with AGP 8.1)
export JAVA_HOME=/home/solaya/.local/java/jdk-17.0.13+11
export PATH="$JAVA_HOME/bin:/home/solaya/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/solaya/miniconda3/bin"

echo "Using Java: $(java -version 2>&1 | head -1)"
echo "Using Cython: $(cython --version 2>&1)"
echo "Building W@Home Hive APK..."

# Run buildozer
buildozer android debug

# Fix gradle.properties for JDK 17 compatibility (if first build)
GRADLE_PROPS=".buildozer/android/platform/build-arm64-v8a/dists/wathome/gradle.properties"
if [ -f "$GRADLE_PROPS" ]; then
    if ! grep -q "org.gradle.jvmargs" "$GRADLE_PROPS"; then
        echo 'org.gradle.jvmargs=-Xmx2048m --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-exports=java.base/sun.nio.ch=ALL-UNNAMED' >> "$GRADLE_PROPS"
    fi
fi

# Copy APK to bin
if ls .buildozer/android/platform/build-arm64-v8a/dists/wathome/build/outputs/apk/debug/*.apk 1>/dev/null 2>&1; then
    cp .buildozer/android/platform/build-arm64-v8a/dists/wathome/build/outputs/apk/debug/*.apk bin/
    echo ""
    echo "APK built successfully:"
    ls -lh bin/*.apk
else
    echo "Build may have failed at Gradle stage. Re-running Gradle directly..."
    cd .buildozer/android/platform/build-arm64-v8a/dists/wathome
    ./gradlew assembleDebug
    cp build/outputs/apk/debug/*.apk "$SCRIPT_DIR/bin/"
    echo ""
    echo "APK built successfully:"
    ls -lh "$SCRIPT_DIR/bin/"*.apk
fi

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist/Build/Products/Release"

echo "=== Building QwenImageApp ==="
cd "$PROJECT_DIR"
xcodebuild build -scheme QwenImageApp -configuration Release -destination 'platform=macOS' -derivedDataPath ./dist -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO

echo ""
echo "=== Creating App Bundle ==="
cd "$BUILD_DIR"
rm -rf QwenImageApp.app dmg-contents QwenImageApp.dmg

mkdir -p QwenImageApp.app/Contents/MacOS
mkdir -p QwenImageApp.app/Contents/Resources

cp QwenImageApp QwenImageApp.app/Contents/MacOS/
chmod +x QwenImageApp.app/Contents/MacOS/QwenImageApp

cp "$PROJECT_DIR/Resources/Info.plist" QwenImageApp.app/Contents/
cp -r mlx-swift_Cmlx.bundle QwenImageApp.app/Contents/Resources/

echo ""
echo "=== App Bundle Created ==="
echo "Location: $BUILD_DIR/QwenImageApp.app"
echo ""
echo "Please sign the app now. When done, press Enter to create DMG..."
read -r

echo ""
echo "=== Creating DMG ==="
mkdir -p dmg-contents
cp -r QwenImageApp.app dmg-contents/
ln -s /Applications dmg-contents/Applications
hdiutil create -volname "Qwen Image" -srcfolder dmg-contents -ov -format UDZO QwenImageApp.dmg

echo ""
echo "=== Done ==="
echo "DMG created: $BUILD_DIR/QwenImageApp.dmg"
ls -lh QwenImageApp.dmg

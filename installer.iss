; W@Home Hive — Inno Setup Installer Script
; One-click installer: WHome.exe (GUI) + WHome.scr (screensaver)
;
; Build:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

[Setup]
AppName=W@Home Hive
AppVersion=1.0.0
AppPublisher=Akataleptos
AppPublisherURL=https://akataleptos.com
AppSupportURL=https://wathome.akataleptos.com
DefaultDirName={autopf}\WHome
DefaultGroupName=W@Home Hive
OutputDir=dist
OutputBaseFilename=WHome-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
UninstallDisplayIcon={app}\WHome.exe
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\WHome.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\WHome.scr"; DestDir: "{sys}"; Flags: ignoreversion uninsrestartdelete

[Icons]
Name: "{group}\W@Home Hive"; Filename: "{app}\WHome.exe"
Name: "{group}\Uninstall W@Home"; Filename: "{uninstallexe}"
Name: "{autodesktop}\W@Home Hive"; Filename: "{app}\WHome.exe"; Tasks: desktopicon
Name: "{autostartup}\W@Home Hive"; Filename: "{app}\WHome.exe"; Tasks: autostart

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "autostart"; Description: "Start W@Home when Windows starts"; GroupDescription: "Startup:"
Name: "screensaver"; Description: "Set W@Home as the default screensaver"; GroupDescription: "Screensaver:"

[Registry]
; Set WHome as the active screensaver (only if user checked the task)
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "SCRNSAVE.EXE"; ValueData: "{sys}\WHome.scr"; Flags: uninsdeletevalue; Tasks: screensaver
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "ScreenSaveActive"; ValueData: "1"; Tasks: screensaver
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "ScreenSaveTimeOut"; ValueData: "300"; Tasks: screensaver

[UninstallRun]
; Kill screensaver and GUI before uninstall so files aren't locked
Filename: "taskkill"; Parameters: "/F /IM WHome.scr"; Flags: runhidden; RunOnceId: "KillScr"
Filename: "taskkill"; Parameters: "/F /IM WHome.exe"; Flags: runhidden; RunOnceId: "KillGui"

[UninstallDelete]
Type: files; Name: "{sys}\WHome.scr"
Type: files; Name: "{app}\worker_config.json"
Type: files; Name: "{app}\checkpoint.json"
Type: files; Name: "{app}\compute_status.json"
Type: dirifempty; Name: "{app}"

[Run]
Filename: "{app}\WHome.exe"; Description: "Launch W@Home Hive"; Flags: nowait postinstall skipifsilent

{ sources, lib, python3Packages }:

python3Packages.buildPythonApplication rec {
  pname = "labelme";
  version = "0.0.0";

  src = sources.labelme;

  doCheck = false;

  propagatedBuildInputs = with python3Packages; [
    matplotlib
    numpy
    pillow
    pyqt5
    pyyaml
    termcolor
  ];

  meta = with lib; {
    inherit (sources.labelme) homepage description;
    license = licenses.gpl3;
  };
}

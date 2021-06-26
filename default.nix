{ sources ? import ./nix/sources.nix
, pkgs ? import ./nix { inherit sources; }
}:
rec {
  cuticle = pkgs.python39Packages.buildPythonPackage {
    pname = "cuticle_analysis";
    version = "0.0.1";

    src = builtins.fetchGit ./.;

    propagatedBuildInputs = with pkgs.python39Packages; [
      click
      gdown
      matplotlib
      numpy
      opencv3
      openpyxl
      pandas
      pillow
      pygame
      rich
      scikit-learn
      scipy
      tensorflow
    ];

    doCheck = false;
    pythonImportsCheck = [ "cuticle_analysis" ];
  };
}

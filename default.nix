{ sources ? import ./nix/sources.nix
, pkgs ? import ./nix { inherit sources; }
}:
with pkgs;
rec {
  cuticle = python39Packages.buildPythonPackage {
    pname = "cuticle_analysis";
    version = "0.0.1";

    src = lib.cleanSourceWith {
      filter = (path: type:
        ! (builtins.any
            (r: (builtins.match r (builtins.baseNameOf path)) != null)
            [
              "dataset"
              "logs"
              "result"
              "pip_packages"
              ".*\.egg-info"
              ".*\.zip"
            ])
      );
      src = lib.cleanSource ./.;
    };

    propagatedBuildInputs = with python39Packages; [
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

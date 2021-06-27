{ sources ? import ./nix/sources.nix
, pkgs ? import ./nix { inherit sources; }
}:
with pkgs;
rec {
  cuticle = poetry2nix.mkPoetryApplication {
    projectDir = ./.;
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

    doCheck = false;
    pythonImportsCheck = [ "cuticle_analysis" ];
  };
}

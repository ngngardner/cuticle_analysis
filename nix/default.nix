{ sources ? import ./sources.nix }:
import sources.nixpkgs {
  overlays = [
    (_: pkgs: { inherit sources; })
    (_: pkgs: { labelme = pkgs.callPackage ./labelme.nix {}; })
  ];
  config = { };
}

{ pkgs ? import <nixpkgs> {} }:

with pkgs;

python3Packages.buildPythonPackage{
    pname = "cuticle_analysis";
    version = "0.0.1";

    src = builtins.fetchGit ./.;

    propagatedBuildInputs = with python3Packages; [
        click
        gdown
        matplotlib
        numpy
        opencv3
        openpyxl
        pandas
        pygame
        pytorch
        rich
        scikit-learn
        scipy
    ];

    doCheck = false;
    pythonImportsCheck = [ "cuticle_analysis" ];
}

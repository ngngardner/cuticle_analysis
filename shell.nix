with import <nixpkgs> { };

mkShell {
    name = "cuticle_analysis_dev_env";

    buildInputs = with python39Packages; [
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
}

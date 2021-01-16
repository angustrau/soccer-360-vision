{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python38
    python38Packages.imutils
    (python38Packages.opencv3.override {
      enableGtk2 = true;
    })
  ];
}

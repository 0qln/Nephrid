{
  description = "Nephrid";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/f4b140d5b253f5e2a1ff4e5506edbf8267724bde";
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [];
      systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }:
        with pkgs.lib; let
          overrides = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
        in {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [];
          };

          devShells.default = with pkgs;
            mkShell {
              packages = [
                cutechess
              ];

              buildInputs = [
                clang
                llvmPackages_21.bintools
                rustup
              ];

              RUSTC_VERSION = overrides.toolchain.channel;

              # https://github.com/rust-lang/rust-bindgen#environment-variables
              LIBCLANG_PATH = makeLibraryPath [llvmPackages_21.libclang.lib];

              shellHook = ''
                export PATH=$PATH:''${CARGO_HOME:-~/.cargo}/bin
                export PATH=$PATH:''${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
              '';

              # Add precompiled library to rustc search path
              RUSTFLAGS = builtins.map (a: ''-L ${a}/lib'') [
                # add libraries here (e.g. pkgs.libvmi)
              ];

              # Add glibc, clang, glib, and other headers to bindgen search path
              BINDGEN_EXTRA_CLANG_ARGS =
                # Includes normal include path
                (builtins.map (a: ''-I"${a}/include"'') [
                  # add dev libraries here (e.g. pkgs.libvmi.dev)
                  glibc.dev
                ])
                # Includes with special directory paths
                ++ [
                  ''-I"${llvmPackages_21.libclang.lib}/lib/clang/${llvmPackages_21.libclang.version}/include"''
                  ''-I"${glib.dev}/include/glib-2.0"''
                  ''-I${glib.out}/lib/glib-2.0/include/''
                ];
            };
        };
      flake = {};
    };
}

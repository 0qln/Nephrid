{
  description = "Nephrid";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";

    nixpkgs.url = "github:NixOS/nixpkgs/master";

    nixpkgs-cuda.url = "github:NixOS/nixpkgs/2fb006b87f04c4d3bdf08cfdbc7fab9c13d94a15";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
        pkgs-cuda,
        system,
        ...
      }:
        with pkgs.lib; let
          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in {
          _module.args.pkgs-cuda = import inputs.nixpkgs-cuda {
            inherit system;
            config = {
              # https://nixos.org/manual/nixpkgs/unstable/#cuda-configuring-nixpkgs-for-cuda
              # https://nixos.wiki/wiki/CUDA
              allowUnfree = true;
              allowBroken = true;
              cudaSupport = false; # some test failing idk
              cudaVersion = "12.8";
              cudaForwardCompat = true;
              cudaCapabilities = [];
            };
          };

          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [(import inputs.rust-overlay)];
          };

          packages = {
            nephrid-nn = rustPlatform.buildRustPackage {
              pname = "nephrid";
              version = "nn";
              src = ./.;
              cargoLock.lockFile = ./Cargo.lock;
              buildNoDefaultFeatures = true;
              buildFeatures = ["mcts-nn" "gpu"];
              nativeBuildInputs = [pkgs.pkg-config];
              buildInputs =
                [pkgs.zlib]
                ++ (with pkgs-cuda; [
                  cudaPackages_12_8.cuda_cudart
                  cudaPackages_12_8.cudnn
                  cudaPackages_12_8.cudatoolkit
                ]);
            };

            nephrid-hce = rustPlatform.buildRustPackage {
              pname = "nephrid";
              version = "hce";
              src = ./.;
              cargoLock.lockFile = ./Cargo.lock;
              buildNoDefaultFeatures = true;
              buildFeatures = ["mcts-hce" "cpu"];
              nativeBuildInputs = [pkgs.pkg-config];
            };

            default = config.packages.nephrid-hce;
          };

          devShells.default = pkgs.mkShell {
            packages =
              (with pkgs; [
                cutechess
                bacon
                perf
                act
                python3
              ])
              ++ (with pkgs-cuda; [
                # reference: https://discourse.nixos.org/t/cuda-12-8-support-in-nixpkgs/60645/39
                autoconf
                binutils
                cudaPackages_12_8.cuda_cudart
                cudaPackages_12_8.cudnn
                cudaPackages_12_8.cudatoolkit
                curl
                freeglut
                git
                gitRepo
                gnumake
                gnupg
                gperf
                libGL
                libGLU
                linuxPackages.nvidia_x11
                m4
                ncurses5
                procps
                stdenv.cc
                unzip
                util-linux
                xorg.libX11
                xorg.libXext
                xorg.libXi
                xorg.libXmu
                xorg.libXrandr
                xorg.libXv
                zlib
              ]);

            buildInputs = [
              rustToolchain
            ];

            shellHook =
              # bash
              (with pkgs-cuda; ''
                # CUDA libraries
                # how to get lib/ of nvrtc: https://github.com/NixOS/nixpkgs/pull/297590/files#diff-59c22b0fc67d897077e55030166ca816d19c80b7767b2ad486bc0aaa2a772115R494
                export LD_LIBRARY_PATH="${lib.getLib cudaPackages.cuda_nvrtc}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
                export CUDA_PATH=${cudatoolkit}
              '')
              +
              # bash
              ''
                # Rust configuration
                export RUST_BACKTRACE=full
                export PROJECT_ROOT="$PWD"
              ''
              +
              # bash
              ''
                # Dev Docs
                echo "yo! o/"
                echo ""
                echo "howto compilation (example): "
                echo '$ cargo run --bin nephrid --release --features "mcts-nn,gpu" --no-default-features'
                echo ""
                echo "to run all tests: "
                echo '$ cargo test --workspace'
              ''
              +
              # bash
              ''
                # PATH
                export PATH="$PWD/benchmarking/bin:$PATH"
                export PATH="$PWD/bin:$PATH"
              '';
          };
        };
      flake = {};
    };
}

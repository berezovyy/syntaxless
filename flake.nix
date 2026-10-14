{
  description = "Syntaxless development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      # `bun fmt` ships only in canary builds so far. This is the build `bun run format` pins with
      # bunx (it reports itself as 1.4.4-canary.1); npm versions are immutable, unlike Bun's rolling
      # GitHub canary release. Move to nixpkgs' bun once a stable release includes `bun fmt`.
      bunVersion = "1.4.3-canary.20261013.1";
      # nixpkgs-unstable no longer supports Intel macOS (x86_64-darwin), so neither does this flake.
      bunTarballs = {
        x86_64-linux = {
          name = "bun-linux-x64";
          hash = "sha512-JBtSRtvA1pSsGXwI6NO6erKwv84WEpnFkCHd18FldqEoRvZAbTYbw4ay1lmqX87yygyA7PKYC/PWtnHwcY8RUQ==";
        };
        aarch64-linux = {
          name = "bun-linux-aarch64";
          hash = "sha512-pVbdtBx1fFyiO6RTanybSc4Ug7mjKj8HZntmf2+dpQtlaVgVVWsfeWdJu3VTGl5AIxV9RWOJnEBttIQOAhDE9Q==";
        };
        aarch64-darwin = {
          name = "bun-darwin-aarch64";
          hash = "sha512-w+peuipolzphHFRM1ClBPj9Woi7KJuK0vSMQ9zlENqyzRoe1Nt3x59+AjnY/IFhDH/ih+n92tuVtQShHgZi7ug==";
        };
      };

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs (builtins.attrNames bunTarballs) (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: {
        bun-canary =
          let
            tarball = bunTarballs.${pkgs.stdenv.hostPlatform.system};
          in
          pkgs.stdenvNoCC.mkDerivation {
            pname = "bun-canary";
            version = bunVersion;
            src = pkgs.fetchurl {
              url = "https://registry.npmjs.org/@oven/${tarball.name}/-/${tarball.name}-${bunVersion}.tgz";
              inherit (tarball) hash;
            };
            sourceRoot = "package";
            nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.autoPatchelfHook ];
            buildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.stdenv.cc.cc.lib ];
            dontConfigure = true;
            dontBuild = true;
            dontStrip = true; # the executable carries Bun's embedded runtime after its ELF/Mach-O sections
            installPhase = ''
              runHook preInstall
              install -Dm755 bin/bun $out/bin/bun
              ln -s bun $out/bin/bunx
              runHook postInstall
            '';
            meta.mainProgram = "bun";
          };
        default = self.packages.${pkgs.stdenv.hostPlatform.system}.bun-canary;
      });

      devShells = forAllSystems (
        pkgs:
        let
          bun = self.packages.${pkgs.stdenv.hostPlatform.system}.bun-canary;
          # PyPI's manylinux wheels (torch, numpy) expect these libraries on the loader path.
          wheelLibs = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux (
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          );
        in
        {
          # Packages, training and every check: `bun run check`, `bun run test:python`.
          default = pkgs.mkShell {
            packages = [
              bun
              pkgs.nodejs_22 # the packed-tarball checks also install into a Node consumer
              pkgs.python313
              pkgs.uv
            ];
            env = {
              UV_PYTHON = "${pkgs.python313}/bin/python3.13";
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
              export LD_LIBRARY_PATH="${wheelLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };

          # README films: docs/media/v3/render.sh.
          media = pkgs.mkShell {
            packages = [
              bun
              pkgs.nodejs_22
              (pkgs.python3.withPackages (ps: [ ps.pillow ]))
              pkgs.ffmpeg # built with libwebp and libx264
            ];
          };
        }
      );

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}

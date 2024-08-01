{ pkgs, ... }: {
  # Which nixpkgs channel to use
  channel = "stable-23.11"; # or "unstable"

  # List of packages to include in the environment
  packages = [
    pkgs.python311
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.scipy
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.jupyter
    pkgs.python311Packages.notebook
    pkgs.git
    pkgs.curl
    pkgs.wget
    pkgs.python312Full
    
  ];

  # Environment variables for the workspace
  env = {
    # Example environment variables
    # PATH = "/usr/local/bin:${env.PATH}";
  };

  idx = {
    # Extensions for VSCode
    extensions = [
      "ms-python.python"
      "ms-toolsai.jupyter"
    ];

    # Enable previews for web applications
    previews = {
      enable = true;
      previews = {
        web = {
          # Example command for previewing a web app
          command = ["npm" "run" "dev"];
          manager = "web";
          env = {
            PORT = "$PORT";
          };
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      onCreate = {
        # Example to install dependencies or open specific files
        default.openFiles = [ ".idx/dev.nix" "README.md" ];
      };

      onStart = {
        # Example to start a background task
        # watch-backend = "npm run watch-backend";
      };
    };
  };
}

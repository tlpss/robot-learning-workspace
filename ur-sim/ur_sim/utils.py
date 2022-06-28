import pathlib


def get_assets_path() -> pathlib.Path:
    current_file_path = pathlib.Path(__file__)
    assets_path = current_file_path.parent / "assets"
    return assets_path


if __name__ == "__main__":
    print(get_assets_path())
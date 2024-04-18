import subprocess


if __name__ == "__main__":
    subprocess.call(
        "isic image download --search 'family_hx_mm:true OR family_hx_mm:false' data/raw/isic/"
    )

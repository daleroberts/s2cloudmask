from setuptools import setup, find_packages

setup(
    version="0.1",
    name="s2cloudmask",
    description="Sentinel-2 Cloud Masking",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    install_requires=["numpy", "xgboost", "scikit-learn", "scikit-image", "joblib", "xarray"],
    url="https://github.com/daleroberts/s2cloudmask",
    packages=find_packages("."),
    package_dir={"": "."},
    package_data={
        "s2cloudmask": ["models/spectral-model.pkl.xz", "models/temporal-model.pkl.xz"]
    },
    zip_safe=False,
)

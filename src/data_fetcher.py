import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sunpy.map
from astropy.io.fits.verify import VerifyWarning
from sunpy.net import Fido, attrs as a


class SunpyFetcher:
    def __init__(self, download_dir: str = "./data/sunpy_images/"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_map(file_path: Path) -> sunpy.map.Map:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=VerifyWarning)
            return sunpy.map.Map(str(file_path))

    def fetch_hmi_continuum(self, target_date_str: str, time_window_sec: int = 20) -> sunpy.map.Map:
        """
        Fetches an HMI continuum image for a given datetime string.
        Time format: %Y-%m-%dT%H:%M:%SZ
        """
        date = datetime.strptime(target_date_str, "%Y-%m-%dT%H:%M:%SZ")
        start_date = date - timedelta(seconds=time_window_sec)
        end_date = date + timedelta(seconds=time_window_sec)
        
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"Searching HMI data between {start_date_str} and {end_date_str}")
        
        instrument = a.Instrument.hmi
        
        # Note: the original used hmi_cont = hmi_data["vso"][hmi_data["vso"]["Physobs"] == "intensity"]
        hmi_data = Fido.search(a.Time(start_date_str, end_date_str), instrument)
        
        try:
            hmi_cont = hmi_data["vso"][hmi_data["vso"]["Physobs"] == "intensity"]
        except Exception as e:
            print(f"Could not filter by Physobs=intensity, fetching all. Detail: {e}")
            hmi_cont = hmi_data

        if len(hmi_cont) == 0:
            raise RuntimeError(
                f"No HMI records found between {start_date_str} and {end_date_str}."
            )

        safe_timestamp = date.strftime("%Y-%m-%dT%H-%M-%SZ")
        cached_file_path = self.download_dir / (
            f"{safe_timestamp}_{instrument.value.replace('.', '_').upper()}_CONTINUUM.fits"
        )

        if cached_file_path.exists():
            print(f"File {cached_file_path} already exists. Validating cache...")
            try:
                return self._load_map(cached_file_path)
            except Exception as exc:
                print(f"Cached file is unreadable ({exc}). Re-downloading...")

        print("Downloading from JSOC/VSO...")
        download_files = Fido.fetch(hmi_cont, path=str(self.download_dir / "{file}"))
        if len(download_files) == 0:
            raise RuntimeError("Download did not return any files.")

        last_error = None
        for downloaded_file in download_files:
            downloaded_path = Path(downloaded_file)
            try:
                amap = self._load_map(downloaded_path)
            except Exception as exc:
                last_error = exc
                continue

            print(f"Using downloaded file: {downloaded_path}")
            return amap

        raise RuntimeError(
            f"Downloaded files are unreadable. Last error: {last_error}"
        )

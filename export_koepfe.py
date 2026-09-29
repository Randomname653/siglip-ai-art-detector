r"""export_koepfe.py — pack a trained head ensemble as a release file (float16, with checksums).

The heads are small MLPs on frozen features; the backbone (e.g. SigLIP 2) is downloaded from Hugging Face
at run time, so the release only contains the heads. Unpack the zip into DETEKTOR_DATEN and run the scanner:

    python export_koepfe.py p8sx            -> dist/p8sx_heads.zip
    python pruefe.py <folder> --kopf p8sx
"""
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesen import DATEN                          # noqa: E402


def main():
    tag = sys.argv[1]
    ziel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
    os.makedirs(ziel, exist_ok=True)
    zp = os.path.join(ziel, f"{tag}_heads.zip")
    info = {"tag": tag, "heads": []}
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        i = 0
        while os.path.exists(os.path.join(DATEN, f"kopf_{tag}_s{i}.npz")):
            d = dict(np.load(os.path.join(DATEN, f"kopf_{tag}_s{i}.npz")))
            for k in ("w1", "b1", "w2", "b2", "mean", "scale"):
                if k in d and d[k].dtype == np.float32:
                    d[k] = d[k].astype(np.float16) if k in ("w1", "w2") else d[k]
            buf = io.BytesIO()
            np.savez_compressed(buf, **d)
            name = f"kopf_{tag}_s{i}.npz"
            z.writestr(name, buf.getvalue())
            info["heads"].append({"file": name, "sha256": hashlib.sha256(buf.getvalue()).hexdigest()})
            if i == 0:
                info.update(backbones=[str(r) for r in d["rueckgrate"]], threshold_1pct=float(d["schwelle_1"]),
                            threshold_5pct=float(d["schwelle_5"]), input_dim=int(d["mean"].shape[0]))
            i += 1
        z.writestr("heads.json", json.dumps(info, indent=1))
    print(f"{i} heads -> {zp} ({os.path.getsize(zp) / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

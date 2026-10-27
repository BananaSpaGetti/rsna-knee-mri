"""Save a new version of a Kaggle notebook WITHOUT running it (Kaggle's "Quick Save").

`kaggle kernels push` always saves and runs. The save request in kagglesdk has a `kernel_execution_type` field
(SAVE_AND_RUN_ALL / INTER_ACTIVE / QUICK_SAVE) that the CLI never sets; this script sets QUICK_SAVE and then calls
the CLI's own push code, so the folder layout (code file + kernel-metadata.json) is the same as for a push.

The new version has no output; earlier versions keep theirs (see the kaggle-competition skill's
kernel_output_version.py). Submissions pin a version number, so they are not affected.

  py notebooks/kaggle_quick_save.py <folder with kernel-metadata.json>
"""
import sys

from kaggle.api import kaggle_api_extended as ext
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

_Orig = ext.ApiSaveKernelRequest


class _QuickSave(_Orig):
    def __init__(self):
        super().__init__()
        self.kernel_execution_type = KernelExecutionType.QUICK_SAVE


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ext.ApiSaveKernelRequest = _QuickSave
    api = ext.KaggleApi()
    api.authenticate()
    r = api.kernels_push(sys.argv[1])
    err = getattr(r, "error", None)
    print("version", getattr(r, "version_number", None), "url", getattr(r, "url", None), "error", err)
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
preprocess_dicom.py

Converts the NIH Pancreas-CT dataset (DICOM series, one folder per patient)
into NIfTI (.nii.gz) volumes and arranges them into the train/test folder
layout expected by dataio/loader/cmr_3D_dataset.py (CMR3DDataset):

    <output_root>/train/image/imageXXXX.nii.gz
    <output_root>/train/label/labelXXXX.nii.gz
    <output_root>/test/image/imageXXXX.nii.gz
    <output_root>/test/label/labelXXXX.nii.gz

CMR3DDataset simply sorts the files in the image/ and label/ folders and
pairs them up by position (see dataio/loader/cmr_3D_dataset.py lines 15-17),
so it is critical that the Nth image file and the Nth label file (in sorted
order) belong to the same patient. We guarantee this by giving every image
the exact same "XXXX" id as its matching label file
(e.g. label0007.nii.gz <-> image0007.nii.gz).

Patient identification:
    The DICOM folder/sub-folder names in TCIA downloads are Series Instance
    UIDs (e.g. "1.2.826.0.1.3680043...") - NOT patient numbers - so they
    cannot be used to match a scan to its label. Instead, we read the real
    PatientID from the DICOM metadata itself (tag 0010,0020, e.g.
    "PANCREAS_0001") and match that against the numeric id embedded in each
    labelXXXX.nii.gz filename. If a patient id can't be read from the DICOM
    files, or can't be matched to exactly one label, the script raises an
    error rather than guessing.

Expected input layout (Google Drive, mounted in Colab):
    /content/drive/MyDrive/pancreas_data/
        <patient_folder_1>/*.dcm      # one DICOM series per patient (may be nested)
        <patient_folder_2>/*.dcm
        ...
        labels/
            label0001.nii.gz
            label0002.nii.gz
            ...

This script only needs to be run once (in Colab, where the dataset lives).
It is NOT executed automatically here - run it yourself with:
    python preprocess_dicom.py

Requirements (install in Colab before running):
    pip install SimpleITK
"""

import argparse
import os
import re
import random
import shutil

import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Pancreas-CT DICOM series to NIfTI and split into train/test')
    parser.add_argument('--dicom_root', type=str,
                         default='/content/drive/MyDrive/pancreas_data',
                         help='Folder containing one sub-folder of .dcm files per patient')
    parser.add_argument('--label_dir', type=str,
                         default='/content/drive/MyDrive/pancreas_data/labels',
                         help='Folder containing labelXXXX.nii.gz ground-truth volumes')
    parser.add_argument('--output_root', type=str,
                         default='/content/drive/MyDrive/pancreas_data_prepared',
                         help='Destination root_dir to be passed to CMR3DDataset (contains train/ and test/)')
    parser.add_argument('--test_fraction', type=float, default=0.2,
                         help='Fraction of patients to hold out for the test split (default 0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                         help='Random seed used to shuffle patients before splitting, for reproducibility')
    return parser.parse_args()


# Matches the numeric id in names like "label0001.nii.gz" or "PANCREAS_0001".
ID_PATTERN = re.compile(r'(\d+)')

# DICOM tag for Patient ID (0010,0020) in SimpleITK's "group|element" string form.
PATIENT_ID_TAG = '0010|0020'


def extract_numeric_id(name):
    """Pull the last run of digits out of a string, e.g. 'PANCREAS_0007' -> '0007'."""
    matches = ID_PATTERN.findall(name)
    if not matches:
        return None
    return matches[-1]


def list_dicom_series_folders(dicom_root, label_dir):
    """Return every immediate sub-folder of dicom_root that holds DICOM slices.

    The labels/ folder itself lives inside dicom_root, so it is excluded.
    Each returned folder is expected to contain (possibly nested) the DICOM
    series for exactly one patient.
    """
    folders = []
    for entry in sorted(os.listdir(dicom_root)):
        full_path = os.path.join(dicom_root, entry)
        if not os.path.isdir(full_path):
            continue
        if os.path.abspath(full_path) == os.path.abspath(label_dir):
            continue
        folders.append(full_path)
    return folders


def find_dicom_series_directory(patient_folder):
    """Locate the actual directory holding the .dcm slice files.

    TCIA DICOM downloads are often nested a few levels deep
    (patient/date-study/series-description/*.dcm), so we walk the tree and
    return the first directory that SimpleITK recognises as a DICOM series,
    along with that series' SeriesInstanceUID.
    """
    for root, _dirs, files in os.walk(patient_folder):
        if any(f.lower().endswith('.dcm') for f in files):
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            if series_ids:
                return root, series_ids[0]
    return None, None


def read_patient_id(series_dir, series_id):
    """Read the real PatientID (DICOM tag 0010,0020) from a series' first slice.

    This is the only reliable way to know which patient a series belongs to -
    folder names are Series Instance UIDs, not patient numbers.
    Returns None if the tag is missing.
    """
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, series_id)
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_names[0])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    if not reader.HasMetaDataKey(PATIENT_ID_TAG):
        return None
    patient_id = reader.GetMetaData(PATIENT_ID_TAG).strip()
    return patient_id or None


def collect_dicom_series(dicom_folders):
    """For every patient folder, locate its DICOM series and read its real PatientID.

    Returns a list of dicts: {'folder', 'series_dir', 'series_id', 'patient_id', 'numeric_id'}.
    Raises a RuntimeError immediately if any folder has no readable series or PatientID,
    since we must never guess a patient's identity.
    """
    series_info = []
    for folder in dicom_folders:
        series_dir, series_id = find_dicom_series_directory(folder)
        if series_dir is None:
            raise RuntimeError('No DICOM series found under {0}'.format(folder))

        patient_id = read_patient_id(series_dir, series_id)
        if patient_id is None:
            raise RuntimeError(
                'Could not read PatientID (DICOM tag {0}) from series under {1}. '
                'Refusing to guess the patient identity from the folder name.'.format(PATIENT_ID_TAG, series_dir)
            )

        numeric_id = extract_numeric_id(patient_id)
        if numeric_id is None:
            raise RuntimeError(
                'PatientID "{0}" (from {1}) has no numeric id that can be matched '
                'against label filenames.'.format(patient_id, series_dir)
            )

        series_info.append({
            'folder': folder,
            'series_dir': series_dir,
            'series_id': series_id,
            'patient_id': patient_id,
            'numeric_id': numeric_id,
        })
    return series_info


def match_patients_to_labels(series_info, label_files):
    """Pair each DICOM series with its label file using the real PatientID's numeric id.

    Every series must match exactly one label and every label must match exactly
    one series - any duplicate or missing match raises a clear error instead of
    silently falling back to positional pairing.
    """
    labels_by_id = {}
    for label_path in label_files:
        label_numeric_id = extract_numeric_id(os.path.basename(label_path))
        if label_numeric_id is None:
            raise RuntimeError('Label file {0} has no numeric id in its name.'.format(label_path))
        if label_numeric_id in labels_by_id:
            raise RuntimeError(
                'Duplicate label id {0}: both {1} and {2} resolve to the same id.'.format(
                    label_numeric_id, labels_by_id[label_numeric_id], label_path
                )
            )
        labels_by_id[label_numeric_id] = label_path

    pairs = []
    seen_ids = {}
    for info in series_info:
        numeric_id = info['numeric_id']
        if numeric_id in seen_ids:
            raise RuntimeError(
                'Duplicate patient id {0}: both {1} (PatientID={2}) and {3} (PatientID={4}) '
                'resolve to the same numeric id.'.format(
                    numeric_id, seen_ids[numeric_id], info['patient_id'], info['folder'], info['patient_id']
                )
            )
        seen_ids[numeric_id] = info['folder']

        label_path = labels_by_id.get(numeric_id)
        if label_path is None:
            raise RuntimeError(
                'No label file found matching PatientID "{0}" (numeric id {1}) from {2}.'.format(
                    info['patient_id'], numeric_id, info['folder']
                )
            )
        pairs.append((info, label_path, numeric_id))

    if len(pairs) != len(label_files):
        matched_ids = {numeric_id for _, _, numeric_id in pairs}
        unmatched_labels = [f for lid, f in labels_by_id.items() if lid not in matched_ids]
        raise RuntimeError(
            'Not every label file was matched to a patient series. '
            'Unmatched labels: {0}'.format(unmatched_labels)
        )
    return pairs


def load_dicom_series_as_sitk_image(series_dir, series_id):
    """Read a full DICOM series (all slices) into a single 3D SimpleITK image."""
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, series_id)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    return reader.Execute()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Step 1: gather the raw inputs - one DICOM folder and one label file per patient.
    dicom_folders = list_dicom_series_folders(args.dicom_root, args.label_dir)
    label_files = sorted(
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir)
        if f.endswith('.nii.gz')
    )
    print('Found {0} DICOM patient folders and {1} label files.'.format(len(dicom_folders), len(label_files)))

    # Step 2: read the real PatientID out of each series' DICOM metadata
    # (folder names are Series Instance UIDs, not patient numbers, so they
    # cannot be trusted to identify the patient).
    series_info = collect_dicom_series(dicom_folders)

    # Step 3: pair every patient's DICOM series with its ground-truth label,
    # using the numeric id embedded in the real PatientID. Raises if any
    # series or label can't be matched exactly - no positional fallback.
    pairs = match_patients_to_labels(series_info, label_files)

    # Step 4: shuffle patients and split them 80/20 into train/test.
    # Shuffling (with a fixed seed) avoids any bias from folder-name ordering.
    random.shuffle(pairs)
    num_test = max(1, round(len(pairs) * args.test_fraction))
    test_pairs = pairs[:num_test]
    train_pairs = pairs[num_test:]
    print('Splitting {0} patients -> {1} train / {2} test.'.format(len(pairs), len(train_pairs), len(test_pairs)))

    # Step 5: create the output folder structure expected by CMR3DDataset:
    #   root_dir/<split>/image/  and  root_dir/<split>/label/
    for split in ('train', 'test'):
        os.makedirs(os.path.join(args.output_root, split, 'image'), exist_ok=True)
        os.makedirs(os.path.join(args.output_root, split, 'label'), exist_ok=True)

    # Step 6: convert each patient's DICOM series to NIfTI and copy its label,
    # writing both into the same split folder with matching "imageXXXX"/"labelXXXX" ids.
    for split, split_pairs in (('train', train_pairs), ('test', test_pairs)):
        image_out_dir = os.path.join(args.output_root, split, 'image')
        label_out_dir = os.path.join(args.output_root, split, 'label')

        for info, label_path, numeric_id in split_pairs:
            print('[{0}] Converting PatientID={1} (numeric id {2}) ...'.format(
                split, info['patient_id'], numeric_id))

            image = load_dicom_series_as_sitk_image(info['series_dir'], info['series_id'])

            # Name the converted image to match its label, e.g. label0007.nii.gz -> image0007.nii.gz,
            # so that sorted(image/) and sorted(label/) line up index-for-index in CMR3DDataset.
            image_filename = 'image{0}.nii.gz'.format(numeric_id)
            image_out_path = os.path.join(image_out_dir, image_filename)
            sitk.WriteImage(image, image_out_path)

            label_filename = os.path.basename(label_path)
            label_out_path = os.path.join(label_out_dir, label_filename)
            shutil.copy2(label_path, label_out_path)

    print('Done. Converted dataset is ready at: {0}'.format(args.output_root))
    print('Pass this path as --root_dir / dataset root to CMR3DDataset.')


if __name__ == '__main__':
    main()

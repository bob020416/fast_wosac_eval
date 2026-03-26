import pickle
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from wosac_fast_eval_tool.scenario_gt_converter import extract_gt_scenario


def iter_tfrecord_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.glob('*') if p.is_file())
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f'Input path does not exist: {input_path}')


def convert_file(tfrecord_path: Path, output_dir: Path) -> int:
    count = 0
    dataset = tf.data.TFRecordDataset(tfrecord_path.as_posix(), compression_type='')
    for raw_record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(raw_record.numpy()))
        gt_dict = extract_gt_scenario(scenario)
        output_path = output_dir / f'{scenario.scenario_id}.pkl'
        with open(output_path, 'wb') as handle:
            pickle.dump(gt_dict, handle)
        count += 1
    return count


def main() -> None:
    parser = ArgumentParser(description='Convert Waymo validation TFRecords into GT pickles for offline WOSAC evaluation.')
    parser.add_argument('input_path', type=Path, help='Waymo validation directory or a single TFRecord file.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for GT scenario pickle files.')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tfrecord_paths = iter_tfrecord_paths(args.input_path.resolve())
    total = 0
    for tfrecord_path in tqdm(tfrecord_paths, desc='Preparing GT'):
        total += convert_file(tfrecord_path, output_dir)

    print(f'prepared {total} GT scenarios into {output_dir}')


if __name__ == '__main__':
    main()

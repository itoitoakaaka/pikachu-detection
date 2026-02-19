"""
pikachu_train.py — ピカチュウ検出モデルの学習スクリプト
Google ColabまたはローカルでYOLOv8を使ってピカチュウを検出するモデルを学習する

使い方:
    # 1. データセットのダウンロード
    python pikachu_train.py download --api_key YOUR_ROBOFLOW_API_KEY

    # 2. モデルの学習
    python pikachu_train.py train --data_dir Pikachu-1 --epochs 50

    # 3. 推論
    python pikachu_train.py predict --model runs/detect/train/weights/best.pt --image test.jpg
"""
import argparse
import os
import sys


def download_dataset(api_key, workspace="gian-b0euc", project_name="pikachu-zxwjc", version_num=1):
    """Roboflowからデータセットをダウンロードする"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("⚠️ roboflowがインストールされていません: pip install roboflow")
        sys.exit(1)

    if not api_key:
        print("❌ API keyを指定してください: --api_key YOUR_KEY")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("yolov8")
    print(f"✅ データセットダウンロード完了: {dataset.location}")
    return dataset


def train_model(data_dir, epochs=50, imgsz=640, batch=16, model_name="yolov8s.pt"):
    """YOLOv8モデルを学習する"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ ultralyticsがインストールされていません: pip install ultralytics")
        sys.exit(1)

    data_yaml = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"❌ data.yamlが見つかりません: {data_yaml}")
        sys.exit(1)

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )
    print("✅ 学習完了")
    return results


def predict(model_path, image_path, conf=0.25):
    """学習済みモデルで推論を行う"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ ultralyticsがインストールされていません: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    for r in results:
        print(f"検出数: {len(r.boxes)}")
        for box in r.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            print(f"  クラス: {cls}, 信頼度: {confidence:.3f}, 座標: {box.xyxy[0].tolist()}")

    # 結果画像を保存
    for r in results:
        im = r.plot()
        import cv2
        output_path = "pikachu_detection_result.jpg"
        cv2.imwrite(output_path, im)
        print(f"✅ 検出結果を保存: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ピカチュウ検出モデル")
    subparsers = parser.add_subparsers(dest="command")

    # ダウンロードコマンド
    dl_parser = subparsers.add_parser("download", help="データセットをダウンロード")
    dl_parser.add_argument("--api_key", type=str, required=True, help="Roboflow API Key")

    # 学習コマンド
    train_parser = subparsers.add_parser("train", help="モデルを学習")
    train_parser.add_argument("--data_dir", type=str, default="Pikachu-1", help="データセットディレクトリ")
    train_parser.add_argument("--epochs", type=int, default=50, help="エポック数")
    train_parser.add_argument("--imgsz", type=int, default=640, help="画像サイズ")
    train_parser.add_argument("--batch", type=int, default=16, help="バッチサイズ")

    # 推論コマンド
    pred_parser = subparsers.add_parser("predict", help="推論を実行")
    pred_parser.add_argument("--model", type=str, required=True, help="モデルファイルパス")
    pred_parser.add_argument("--image", type=str, required=True, help="入力画像パス")
    pred_parser.add_argument("--conf", type=float, default=0.25, help="信頼度閾値")

    args = parser.parse_args()

    if args.command == "download":
        download_dataset(args.api_key)
    elif args.command == "train":
        train_model(args.data_dir, args.epochs, args.imgsz, args.batch)
    elif args.command == "predict":
        predict(args.model, args.image, args.conf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

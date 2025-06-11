from ultralytics import YOLO



models = ["n", "s", "m", "l", "x"]

for x in models:
    # Load a model
    model = YOLO(f"yolov8{x}-cls.pt")  # load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(data="/mmfs1/scratch/utsha.saha/skncncr/data/", epochs=30, imgsz=640, project='./result', name=f'yolov8{x}', device=[0,1])
    
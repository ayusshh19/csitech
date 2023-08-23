from ultralytics import YOLO

model = YOLO('C:\\Users\\AYUSH SHUKLA\\Desktop\\v8\\ball\\runs\\detect\\train29\\weights\\last.pt')
model.resume = True 

# train the model
results = model.train(
    epochs=10 # number of additional epochs you want to train on
)
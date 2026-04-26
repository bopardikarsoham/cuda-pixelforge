NVCC = nvcc
TARGET = batch_processor

all: build

build:
	$(NVCC) src/batch_image_processor.cu -o $(TARGET)

run:
	./$(TARGET) input output

clean:
	rm -f $(TARGET)
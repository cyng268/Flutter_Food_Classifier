import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

const int WIDTH = 224;
const int HEIGHT = 224;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Food Classification',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  String? _prediction;
  Interpreter? _interpreter;
  Map<int, String>? _labels; // Changed to Map<int, String>

  @override
  void initState() {
    super.initState();
    loadModel();
    loadLabels();
  }

  Future<void> loadLabels() async {
    _labels = {};
    String labelsData = await DefaultAssetBundle.of(context)
        .loadString('assets/labelmap.csv');
    
    for (String line in labelsData.split('\n')) {
      if (line.trim().isNotEmpty) {
        List<String> parts = line.split(',');
        if (parts.length == 2) {
          int index = int.parse(parts[0]);
          String label = parts[1].trim();
          _labels![index] = label;
        }
      }
    }
  }

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/1.tflite');
  }

  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage = img.copyResize(inputImage, width: WIDTH, height: HEIGHT);
    
    // Get flattened list of RGB values
    List<int> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toInt())
        .toList();
    
    // Convert to Uint8List
    Uint8List uint8Array = Uint8List.fromList(flattenedList);
    
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    
    // Create the final reshaped array
    Uint8List reshapedArray = Uint8List(1 * height * width * channels);
    
    // Rearrange the data
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] = uint8Array[c * height * width + h * width + w];
        }
      }
    }
    
    return reshapedArray.reshape([1, 224, 224, 3]);
  }

  Future<void> classifyImage() async {
  if (_image == null || _interpreter == null) return;

  // Read and decode the image
  var imageData = await _image!.readAsBytes();
  img.Image? decodedImage = img.decodeImage(imageData);
  
  if (decodedImage == null) return;

  // Process the image
  var input = imageToArray(decodedImage);
  
  // Create output tensor with 2024 classes
  var output = Uint8List(1 * 2024).reshape([1, 2024]);

  // Run inference
  _interpreter?.run(input, output);

  // Dequantize output
  const double scale = 0.00390625;
  const int zeroPoint = 0;
  
  List<double> dequantizedOutput = List.generate(output[0].length, (i) {
    int quantizedValue = output[0][i];
    return scale * (quantizedValue - zeroPoint);
  });

  // Create list of indices and probabilities
  List<MapEntry<int, double>> indexedProbs = List.generate(
    dequantizedOutput.length,
    (index) => MapEntry(index, dequantizedOutput[index])
  );

  // Sort by probability in descending order
  indexedProbs.sort((a, b) => b.value.compareTo(a.value));

  // Get top 5
  List<MapEntry<int, double>> top5 = indexedProbs.take(5).toList();

  // Build prediction string
  String predictionText = '';
  for (var entry in top5) {
    String label = _labels?.containsKey(entry.key) ?? false 
        ? _labels![entry.key]! 
        : 'Unknown Class ${entry.key}';
    String probability = (entry.value * 100).toStringAsFixed(2);
    predictionText += '$label: $probability%\n';
  }

  setState(() {
    _prediction = predictionText;
  });
}

  Future<void> pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      }
    });

    if (_image != null) {
      await classifyImage();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Food Classification'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? Text('No image selected.')
                : Image.file(_image!, height: 200),
            SizedBox(height: 16),
            _prediction != null
    ? Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Text(
              'Top 5 Predictions:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              _prediction!,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      )
    : Text('No prediction available.'),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: pickImage,
              child: Text('Select Image'),
            ),
          ],
        ),
      ),
    );
  }
}
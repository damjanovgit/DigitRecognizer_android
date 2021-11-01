package com.dprod.digitrecognizer;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentContainerView;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Path;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.Toast;

import com.dprod.digitrecognizer.ml.Mnist;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import processing.android.CompatUtils;
import processing.android.PFragment;
import processing.core.PApplet;
import processing.core.PImage;

import static processing.core.PConstants.ALPHA;
import static processing.core.PConstants.GRAY;

public class MainActivity extends AppCompatActivity {

    private Button clearBtn;
    private Button predictBtn;

    private boolean clearScreen = false;
    private boolean getImage = false;

    private Context context;

    PImage img;

    public class Sketch extends PApplet {
        public void settings() {
            size(600, 600);
        }

        public void setup() {
            background(0x00);
            stroke(0xffffffff);
            strokeWeight(20);
        }

        public void draw() {
            if(getImage == true){
                img = get();
                //System.out.println("Pixel height: " + img.pixelHeight);
                getImage = false;
            }
            if(clearScreen == true){
                background(0x00);
                clearScreen = false;
            }
            if (mousePressed) {
                line(pmouseX, pmouseY, mouseX, mouseY);
            }
        }
    }

    private PApplet sketch;

    private static ByteBuffer floatArrayToByteArray(float[] input)
    {
        final ByteBuffer buffer = ByteBuffer.allocate(Float.BYTES * input.length);
        buffer.asFloatBuffer().put(input);
        return buffer;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        context = this.getApplicationContext();

        clearBtn = (Button)findViewById(R.id.clearBtn);
        clearBtn.setOnClickListener(v -> clearScreen = true);

        predictBtn = (Button)findViewById(R.id.predictBtn);
        predictBtn.setOnClickListener(v -> {
            getImage = true;
            Thread classifyThread = new Thread(){
                public void run() {
                    while(getImage == true);
                    System.out.println("Velicina: " + img.pixelHeight + " " + img.pixelWidth);
                    img.loadPixels();
                    img.filter(GRAY);
                    img.resize(28,28);
                    //for(int i = 0; i < 28 * 28; i ++)
                    //    img.pixels[i] /= 255.0;

                    float inputArray[] = new float[28 * 28];

                    System.out.println("Nakon resize velicina: " + img.pixelHeight + " " + img.pixelWidth);
                    System.out.println("Values: ");
                    for(int i = 0; i < 28; i++){
                        for(int j = 0; j < 28; j++) {
                            if (img.pixels[i + j * 28] < -10000000) inputArray[i + j * 28] = 0;
                            else inputArray[i + j * 28] = 1;
                            System.out.print(inputArray[i + j * 28] + " ");
                        }
                        System.out.println();
                    }

                    try {
                        Mnist model = Mnist.newInstance(context);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(floatArrayToByteArray(inputArray));

                        // Runs model inference and gets result.
                        Mnist.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        float out[] = outputFeature0.getFloatArray();
                        for(int i = 0; i < out.length; i ++)
                            System.out.println(out[i]);

                        int maxAt = 0;

                        for (int i = 0; i < out.length; i++) {
                            maxAt = out[i] > out[maxAt] ? i : maxAt;
                        }
                        System.out.println("Max na " + maxAt);

                        int a = maxAt;
                        runOnUiThread(new Runnable() {
                            public void run() {
                                Toast.makeText(MainActivity.this, "To je broj " + a, Toast.LENGTH_SHORT).show();
                            }
                        });

                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }
                }
            };
            classifyThread.start();
        });

        FragmentContainerView fcv = findViewById(R.id.fragmentContainerView);

        //FrameLayout frame = new FrameLayout(this);
        //frame.setId(CompatUtils.getUniqueViewId());
        //setContentView(fcv,new ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
        //        ViewGroup.LayoutParams.WRAP_CONTENT));

        sketch = new Sketch();
        PFragment fragment = new PFragment(sketch);
        fragment.setView(fcv, this);
    }

    @Override
    public void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        if (sketch != null) {
            sketch.onNewIntent(intent);
        }
    }
}
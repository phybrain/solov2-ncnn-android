package com.tencent.ncnnhair;

import android.app.Activity;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;

import java.io.File;

public class ImageActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);
        ImageView imageView = (ImageView) findViewById(R.id.img);
        imageView.setImageURI(Uri.fromFile(new File("/storage/emulated/0/DCIM/warp.jpg")));


    }
}
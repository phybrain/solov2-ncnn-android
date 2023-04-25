package com.tencent.ncnnhair;

public class Test {
    private NcnnHair ncnnhair = new NcnnHair();
    private static void callStaticMethod(String str, int i) {
        System.out.format("ClassMethod::callStaticMethod called!-->str=%s," +
                " i=%d\n", str, i);

    }

    private void callInstanceMethod(String str, int i) {
        System.out.format("ClassMethod::callInstanceMethod called!-->str=%s, " +
                "i=%d\n", str, i);
        ncnnhair.closeCamera();
    }
}

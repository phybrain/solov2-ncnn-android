package com.tencent.ncnnhair;

public class DemoThread extends Thread {
    private final OnResultCallback callback;
    public DemoThread(OnResultCallback callback) {
        this.callback = callback;
    }
    @Override
    public void run() {
        super.run();
        //当多线程中操作完成后可在此处回调想要的结果
        while (!NcnnHair.getInvoice()){
            try {
                sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
                callback.onError(0);
            }
        }
        if (callback != null) {
            callback.onComplete(null);
        }

    }
    public interface OnResultCallback {
        void onComplete(Object object);
        void onError(int code);
    }
}
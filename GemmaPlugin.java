package com.neet.pyq;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;

import androidx.activity.result.ActivityResult;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.google.mediapipe.tasks.genai.llminference.LlmInference;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@CapacitorPlugin(name = "Gemma")
public class GemmaPlugin extends Plugin {

    private LlmInference llm;
    private final ExecutorService ex = Executors.newSingleThreadExecutor();

    private File modelFile() {
        return new File(getContext().getFilesDir(), "gemma3-1b.task");
    }

    private synchronized void ensure() throws Exception {
        if (llm != null) return;
        File f = modelFile();
        if (!f.exists()) throw new Exception("Model file not added yet");
        LlmInference.LlmInferenceOptions o = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(f.getAbsolutePath())
                .setMaxTokens(1536)
                .setPreferredBackend(LlmInference.Backend.CPU)
                .build();
        llm = LlmInference.createFromOptions(getContext(), o);
    }

    @PluginMethod
    public void status(PluginCall call) {
        File f = modelFile();
        JSObject o = new JSObject();
        boolean ok = f.exists() && f.length() > 1000000;
        o.put("installed", ok);
        o.put("sizeMB", (int) (f.length() / 1048576));
        o.put("loaded", llm != null);
        call.resolve(o);
    }

    @PluginMethod
    public void pickModel(PluginCall call) {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        startActivityForResult(call, i, "pickResult");
    }

    @ActivityCallback
    private void pickResult(PluginCall call, ActivityResult result) {
        if (call == null) return;
        if (result.getResultCode() != Activity.RESULT_OK || result.getData() == null || result.getData().getData() == null) {
            call.reject("Cancelled");
            return;
        }
        final Uri uri = result.getData().getData();
        ex.execute(() -> {
            try {
                File out = modelFile();
                File tmp = new File(out.getPath() + ".tmp");
                try (InputStream in = getContext().getContentResolver().openInputStream(uri);
                     OutputStream os = new FileOutputStream(tmp)) {
                    if (in == null) throw new IOException("Cannot open file");
                    byte[] buf = new byte[1 << 20];
                    int n;
                    while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
                }
                synchronized (GemmaPlugin.this) {
                    if (llm != null) { llm.close(); llm = null; }
                }
                if (out.exists()) out.delete();
                if (!tmp.renameTo(out)) throw new IOException("Could not save model");
                JSObject o = new JSObject();
                o.put("sizeMB", (int) (out.length() / 1048576));
                call.resolve(o);
            } catch (Throwable e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    @PluginMethod
    public void removeModel(PluginCall call) {
        ex.execute(() -> {
            try {
                synchronized (GemmaPlugin.this) {
                    if (llm != null) { llm.close(); llm = null; }
                }
                File f = modelFile();
                if (f.exists()) f.delete();
                call.resolve();
            } catch (Throwable e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    @PluginMethod
    public void generate(PluginCall call) {
        final String prompt = call.getString("prompt");
        if (prompt == null) { call.reject("No prompt"); return; }
        ex.execute(() -> {
            try {
                ensure();
                String r = llm.generateResponse(prompt);
                JSObject o = new JSObject();
                o.put("text", r == null ? "" : r);
                call.resolve(o);
            } catch (Throwable e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }
            }

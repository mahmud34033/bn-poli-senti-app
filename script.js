import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// ১. অফলাইন সাপোর্ট ক্যাশিং কনফিগারেশন
env.allowLocalModels = true;
env.allowRemoteModels = false;
// এটি অবশ্যই true থাকতে হবে যাতে একবার ডাউনলোড হলে ব্রাউজার সেটা মনে রাখে
env.useBrowserCache = true;
// env.useBrowserCache = false;
env.localModelPath = '';

let classifier;

// DOM Elements
const loadingSection = document.getElementById('loading-section');
const mainInterface = document.getElementById('main-interface');
const statusText = document.getElementById('status');
const inputText = document.getElementById('inputText');
const labelOut = document.getElementById('label');
const scoreOut = document.getElementById('score');
const bar = document.getElementById('bar');
const resetBtn = document.getElementById('resetBtn');
const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');
const explainBtn = document.getElementById('explainBtn');
const limeOutput = document.getElementById('lime-output');
const highlightedText = document.getElementById('highlighted-text');

// ১. ডার্ক মোড লজিক
toggleSwitch.addEventListener('change', (e) => {
    if (e.target.checked) {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.getElementById('mode-text').textContent = "লাইট মোড";
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
        document.getElementById('mode-text').textContent = "ডার্ক মোড";
    }
});

// ২. মডেল লোড
async function init() {
    try {
        // ইউজারকে মডেল লোড এবং অফলাইন ফিচারের তথ্য দেখানো
        statusText.innerHTML = "আমাদের AI মডেলটি আপনার ব্রাউজারে লোড হচ্ছে। এটি মাত্র একবারই ডাউনলোড হবে, এরপর থেকে আপনি অফলাইনেও এটি ব্যবহার করতে পারবেন।<br><br><span style='font-weight:bold; color:var(--primary-color);'>মডেল চেক করা হচ্ছে...</span>";
        
        classifier = await pipeline('sentiment-analysis', './quantized_model', {
            model_file: 'model_quantized.onnx',
            quantized: true,
            cache_dir: 'models-cache',
            progress_callback: (info) => {
                if (info.status === 'progress') {
                    statusText.innerHTML = `আমাদের AI মডেলটি আপনার ব্রাউজারে লোড হচ্ছে। এটি মাত্র একবারই ডাউনলোড হবে, এরপর থেকে আপনি অফলাইনেও এটি ব্যবহার করতে পারবেন।<br><br><b style='color:#3498db;'>ডাউনলোড হচ্ছে: ${info.progress.toFixed(1)}%</b>`;
                }
            }
        });
        loadingSection.classList.add('hidden');
        mainInterface.classList.remove('hidden');
    } catch (err) {
        statusText.textContent = "Error: " + err.message;
    }
}

// ৩. বিশ্লেষণ
async function analyze() {
    const text = inputText.value.trim();
    if (text.length === 0) {
        labelOut.textContent = "অপেক্ষমান";
        labelOut.style.color = 'inherit';
        scoreOut.textContent = "0%";
        bar.style.width = "0%";
        document.getElementById('pos-val').textContent = "0%";
        document.getElementById('neg-val').textContent = "0%";
        document.getElementById('neu-val').textContent = "0%";
        document.getElementById('explanation-section').classList.add('hidden');
        return;
    }

    const output = await classifier(text, { topk: 3 });
    const mainResult = output[0];
    labelOut.textContent = mainResult.label;
    const percentage = (mainResult.score * 100).toFixed(1);
    scoreOut.textContent = percentage + "%";
    bar.style.width = percentage + "%";

    output.forEach(res => {
        const val = (res.score * 100).toFixed(1) + "%";
        const label = res.label.toLowerCase();
        if (label.includes('pos')) document.getElementById('pos-val').textContent = val;
        else if (label.includes('neg')) document.getElementById('neg-val').textContent = val;
        else if (label.includes('neu')) document.getElementById('neu-val').textContent = val;
    });

    // কালার কোড সেট করা
    if (mainResult.label.toLowerCase().includes('pos')) {
        bar.style.backgroundColor = '#2ecc71';
        labelOut.style.color = '#2ecc71';
    } else if (mainResult.label.toLowerCase().includes('neg')) {
        bar.style.backgroundColor = '#e74c3c';
        labelOut.style.color = '#e74c3c';
    } else {
        bar.style.backgroundColor = '#f1c40f';
        labelOut.style.color = '#f1c40f';
    }

    if (text.length > 0) {
        document.getElementById('explanation-section').classList.remove('hidden');
        limeOutput.style.display = 'none';
    }
}

// ৪. রিসেট লজিক
resetBtn.addEventListener('click', () => {
    inputText.value = "";
    analyze();
    if (limeOutput) {
        limeOutput.style.display = 'none';
        highlightedText.innerHTML = "";
    }
});

// ৫. ডায়নামিক কালার এবং টপ ৫ হাইলাইট সহ LIME ক্যালকুলেশন
explainBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    const words = text.split(/\s+/);
    limeOutput.style.display = 'block';
    highlightedText.innerHTML = "বিশ্লেষণ করা হচ্ছে...";

    const originalOutput = await classifier(text);
    const baseScore = originalOutput[0].score;
    const baseLabel = originalOutput[0].label;
    const baseLabelLower = baseLabel.toLowerCase();

    let baseColorRgb;
    if (baseLabelLower.includes('pos')) baseColorRgb = "46, 204, 113"; 
    else if (baseLabelLower.includes('neg')) baseColorRgb = "231, 76, 60"; 
    else baseColorRgb = "241, 196, 15"; 

    let wordImportance = [];

    for (let i = 0; i < words.length; i++) {
        const modifiedWords = [...words];
        modifiedWords.splice(i, 1);
        const modifiedText = modifiedWords.join(' ');

        const newOutput = await classifier(modifiedText);
        // মেইন লেবেলের স্কোর কতটুকু কমলো তা বের করা
        const newScore = newOutput[0].label === baseLabel ? newOutput[0].score : (1 - newOutput[0].score);
        
        const impact = baseScore - newScore;
        wordImportance.push({ index: i, word: words[i], impact: impact });
    }

    // ১. শুধুমাত্র পজিটিভ ইমপ্যাক্ট আছে এমন শব্দগুলো ফিল্টার করা
    const contributingWords = wordImportance.filter(item => item.impact > 0);

    // ২. ইমপ্যাক্ট অনুযায়ী বড় থেকে ছোট সর্ট করা
    const topFive = contributingWords
        .sort((a, b) => b.impact - a.impact)
        .slice(0, 5);

    const topIndices = topFive.map(item => item.index);
    
    // ৩. বড় সেন্টেন্সে হাইলাইট নিশ্চিত করতে 'ম্যাক্সিমাম ইমপ্যাক্ট' দিয়ে নরমালইজ করা
    const maxImpact = topFive.length > 0 ? topFive[0].impact : 1;

    highlightedText.innerHTML = wordImportance.map(item => {
        if (topIndices.includes(item.index)) {
            let relativeOpacity = (item.impact / maxImpact) * 0.7; 
            let opacity = Math.max(0.3, relativeOpacity); 
            
            // এখানে color: #000 (কালো) দিলে হালকা ব্যাকগ্রাউন্ডে লেখা ভালো ফুটবে
            // অথবা কালার ডাইনামিক করতে চাইলে var(--text-main) দিতে পারেন
            let textColor = "#000000"; 
            
            return `<span style="background-color: rgba(${baseColorRgb}, ${opacity}); color: ${textColor}; padding: 3px 8px; border-radius: 6px; margin: 0 3px; font-weight: bold; display: inline-block; border: 1px solid rgba(${baseColorRgb}, 0.3);">
                ${item.word}
            </span>`;
        } else {
            // সাধারণ শব্দের জন্য CSS ক্লাস ব্যবহার করা হচ্ছে
            return `<span class="normal-word" style="margin: 0 2px;">${item.word}</span>`;
        }
    }).join(' ');
});

inputText.addEventListener('input', analyze);
init();
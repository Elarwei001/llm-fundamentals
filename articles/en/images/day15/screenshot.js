const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {
  const htmlFile = path.join(__dirname, 'ppo-memory-problem.html');
  const html = fs.readFileSync(htmlFile, 'utf8');
  
  // Wrap in fixed-size container
  const wrappedHtml = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 1600px; height: 950px; background: #0f172a; overflow: hidden; }
  body > svg { width: 1600px; height: 950px; }
</style></head><body>${html.match(/<svg[\s\S]*<\/svg>/)[0]}</body></html>`;
  
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox', '--disable-gpu']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 950, deviceScaleFactor: 2 });
  await page.setContent(wrappedHtml, { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: path.join(__dirname, 'ppo-memory-problem.png'), fullPage: false });
  await browser.close();
  console.log('done');
})();

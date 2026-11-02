// Vote endpoint for Mask Vote. Paste into the Google Sheet's Extensions > Apps Script,
// set the script property PASSWORD (Project Settings > Script properties), then
// Deploy > New deployment > Web app, execute as: Me, who has access: Anyone.
//   GET  ?pw=...                         -> {"ok": true|false}   (the page's password check)
//   POST {pw, voter, device, item, batch, vote}  -> appends a row to the "votes" sheet

const SHEET = "votes";
const HEADER = ["time", "item", "vote", "voter", "device", "batch"];

function ok_(pw) {
  return pw && pw === PropertiesService.getScriptProperties().getProperty("PASSWORD");
}

function json_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

function doGet(e) {
  return json_({ ok: !!ok_(e.parameter.pw) });
}

function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  if (!ok_(d.pw) || !["up", "down"].includes(d.vote)) return json_({ ok: false });
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const book = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = book.getSheetByName(SHEET) || book.insertSheet(SHEET);
    if (sheet.getLastRow() === 0) sheet.appendRow(HEADER);
    sheet.appendRow([new Date(), d.item, d.vote, String(d.voter).slice(0, 80), d.device, d.batch]);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true });
}

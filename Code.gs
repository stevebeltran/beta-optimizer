function doGet(e) {
  var SPREADSHEET_ID = "1Qga6qzuoRS-BfT-0FE_RgfnX1lFpxDo1gBa55f86oUE";
  var SHEET_NAME = "QR Scans";
  var SOURCE_APP = "BRINC QR Web App";

  var params = (e && e.parameter) ? e.parameter : {};
  var headers = (e && e.headers) ? e.headers : {};
  var queryString = String(e && e.queryString ? e.queryString : "");
  var requestUrl = String(e && e.url ? e.url : "");

  var timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyy-MM-dd HH:mm:ss");
  var reportId = String(params.report_id || params.public_report || "");
  var publicReport = String(params.public_report || "");
  var sig = String(params.sig || "");
  var city = String(params.city || "");
  var state = String(params.state || "");
  var department = String(params.department || "");
  var repName = String(params.rep_name || "");
  var repEmail = String(params.rep_email || "");
  var brincUser = String(params.brinc_user || "");
  var userAgent = String(headers["User-Agent"] || headers["user-agent"] || "");
  var ipAddress = String(
    headers["X-Forwarded-For"] ||
    headers["x-forwarded-for"] ||
    headers["Remote-Addr"] ||
    headers["remote-addr"] ||
    ""
  ).split(",")[0].trim();
  var language = String(headers["Accept-Language"] || headers["accept-language"] || "");
  var host = String(headers["Host"] || headers["host"] || "");
  var referer = String(headers["Referer"] || headers["referer"] || "");
  var origin = String(headers["Origin"] || headers["origin"] || "");
  var accept = String(headers["Accept"] || headers["accept"] || "");
  var paramsJson = safeStringify_(params);
  var headersJson = safeStringify_(headers);
  var device = detectDevice_(String(userAgent || "").toLowerCase());

  var ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  var sheet = ss.getSheetByName(SHEET_NAME);

  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
    sheet.appendRow([
      "Timestamp",
      "Source App",
      "Report ID",
      "Public Report",
      "Signature",
      "City",
      "State",
      "Department",
      "Rep Name",
      "Rep Email",
      "Brinc User",
      "Query String",
      "Request URL",
      "Device",
      "User Agent",
      "Language",
      "IP Address",
      "Host",
      "Referer",
      "Origin",
      "Accept",
      "Params JSON",
      "Headers JSON"
    ]);
  }

  sheet.appendRow([
    timestamp,
    SOURCE_APP,
    reportId,
    publicReport,
    sig,
    city,
    state,
    department,
    repName,
    repEmail,
    brincUser,
    queryString,
    requestUrl,
    device,
    userAgent,
    language,
    ipAddress,
    host,
    referer,
    origin,
    accept,
    paramsJson,
    headersJson
  ]);

  var html = buildLandingPage_({
    timestamp: timestamp,
    reportId: reportId,
    publicReport: publicReport,
    sig: sig,
    city: city,
    state: state,
    department: department,
    repName: repName,
    repEmail: repEmail,
    brincUser: brincUser,
    device: device,
    sourceApp: SOURCE_APP,
    sourceUrl: requestUrl
  });

  return HtmlService
    .createHtmlOutput(html)
    .setTitle(city || "BRINC DFR")
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

function detectDevice_(userAgentLower) {
  if (!userAgentLower) return "";
  if (userAgentLower.indexOf("iphone") >= 0 || userAgentLower.indexOf("ipad") >= 0) return "iOS";
  if (userAgentLower.indexOf("android") >= 0) return "Android";
  if (userAgentLower.indexOf("mobile") >= 0) return "Mobile";
  return "Desktop";
}

function safeStringify_(value) {
  try {
    return JSON.stringify(value);
  } catch (err) {
    return "";
  }
}

function escapeHtml_(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildLandingPage_(data) {
  var city = escapeHtml_(data.city);
  var state = escapeHtml_(data.state);
  var department = escapeHtml_(data.department);
  var repName = escapeHtml_(data.repName || "BRINC Representative");
  var repEmail = String(data.repEmail || "").trim();
  var repEmailHtml = repEmail ? '<a href="mailto:' + escapeHtml_(repEmail) + '">' + escapeHtml_(repEmail) + '</a>' : "&mdash;";
  var location = [city, state].filter(Boolean).join(", ") || "your jurisdiction";
  var reportId = escapeHtml_(data.reportId || "");
  var publicReport = escapeHtml_(data.publicReport || "");
  var device = escapeHtml_(data.device || "");
  var timestamp = escapeHtml_(data.timestamp || "");
  var sourceApp = escapeHtml_(data.sourceApp || "");
  var sourceUrl = String(data.sourceUrl || "").trim();
  var mailto = repEmail ? 'mailto:' + repEmail : "";

  return [
    '<!doctype html>',
    '<html lang="en">',
    '<head>',
    '  <meta charset="utf-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1">',
    '  <meta name="color-scheme" content="dark">',
    '  <title>BRINC DFR - ' + location + '</title>',
    '  <style>',
    '    :root {',
    '      --bg: #07101c;',
    '      --panel: rgba(10, 20, 36, 0.96);',
    '      --panel-soft: rgba(12, 28, 48, 0.82);',
    '      --accent: #00d2ff;',
    '      --accent-soft: #78f0ff;',
    '      --text: #eff6ff;',
    '      --muted: #95a7bd;',
    '      --border: rgba(0, 210, 255, 0.18);',
    '      --shadow: 0 24px 56px rgba(0, 0, 0, 0.34);',
    '    }',
    '    * { box-sizing: border-box; }',
    '    body { margin: 0; min-height: 100vh; font-family: Arial, Helvetica, sans-serif; background: radial-gradient(circle at top, rgba(0, 210, 255, 0.12), transparent 36%), linear-gradient(180deg, #06101b 0%, #07101c 100%); color: var(--text); }',
    '    .shell { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 28px; }',
    '    .card { width: min(920px, 100%); background: var(--panel); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); overflow: hidden; position: relative; }',
    '    .card:before { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,210,255,0.08), transparent 35%); pointer-events: none; }',
    '    .hero { padding: 34px 34px 26px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); }',
    '    .eyebrow { color: var(--accent); font-size: 12px; letter-spacing: 0.24em; text-transform: uppercase; font-weight: 800; margin-bottom: 10px; }',
    '    h1 { margin: 0; font-size: clamp(34px, 5vw, 56px); line-height: 1.05; }',
    '    .subtitle { margin-top: 10px; color: var(--accent-soft); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; font-size: 13px; }',
    '    .location { margin-top: 8px; color: var(--muted); font-size: 16px; }',
    '    .body { padding: 30px 34px 34px; display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; }',
    '    .panel { grid-column: span 12; background: var(--panel-soft); border: 1px solid rgba(255,255,255,0.05); border-radius: 20px; padding: 20px; }',
    '    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }',
    '    .meta { padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); }',
    '    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }',
    '    .value { font-size: 15px; line-height: 1.45; word-break: break-word; }',
    '    .cta-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }',
    '    .btn { display: inline-flex; align-items: center; justify-content: center; min-height: 46px; padding: 0 18px; border-radius: 999px; text-decoration: none; font-weight: 800; letter-spacing: 0.02em; }',
    '    .btn-primary { background: #f7fbff; color: #07101c; }',
    '    .btn-secondary { border: 1px solid rgba(255,255,255,0.14); color: var(--text); background: transparent; }',
    '    .note { margin-top: 14px; color: var(--muted); font-size: 14px; line-height: 1.6; }',
    '    .fineprint { margin-top: 14px; color: #6f8094; font-size: 12px; line-height: 1.5; }',
    '    @media (max-width: 720px) {',
    '      .shell { padding: 14px; }',
    '      .hero, .body { padding-left: 18px; padding-right: 18px; }',
    '      .grid { grid-template-columns: 1fr; }',
    '    }',
    '  </style>',
    '</head>',
    '<body>',
    '  <main class="shell">',
    '    <section class="card">',
    '      <header class="hero">',
    '        <div class="eyebrow">Drone as a First Responder</div>',
    '        <h1>' + (city || "BRINC DFR") + '</h1>',
    '        <div class="subtitle">DFR Deployment Proposal</div>',
    '        <div class="location">' + location + '</div>',
    '      </header>',
    '      <div class="body">',
    '        <section class="panel" style="grid-column: span 12;">',
    '          <div class="grid">',
    '            <div class="meta"><div class="label">Status</div><div class="value">Scan logged to the QR sheet and proposal view opened.</div></div>',
    '            <div class="meta"><div class="label">Representative</div><div class="value">' + repName + '</div></div>',
    '            <div class="meta"><div class="label">Email</div><div class="value">' + repEmailHtml + '</div></div>',
    '            <div class="meta"><div class="label">Department</div><div class="value">' + (department || "&mdash;") + '</div></div>',
    '            <div class="meta"><div class="label">Report ID</div><div class="value">' + (reportId || publicReport || "&mdash;") + '</div></div>',
    '            <div class="meta"><div class="label">Device</div><div class="value">' + (device || "&mdash;") + '</div></div>',
    '          </div>',
    '          <div class="note">This scan has been recorded in Google Sheets. Use the contact link below to follow up, or scan again if you want the record refreshed.</div>',
    '          <div class="cta-row">',
    (repEmail ? '            <a class="btn btn-primary" href="' + escapeHtml_(mailto) + '">Email representative</a>' : ''),
    '            <a class="btn btn-secondary" href="' + escapeHtml_(sourceUrl || "#") + '">Reload page</a>',
    '          </div>',
    '          <div class="fineprint">Logged at ' + timestamp + ' via ' + sourceApp + '.</div>',
    '        </section>',
    '      </div>',
    '    </section>',
    '  </main>',
    '</body>',
    '</html>'
  ].join("\n");
}

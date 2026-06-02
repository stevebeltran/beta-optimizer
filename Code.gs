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

function lookupCoordinates_(city, state) {
  var query = [String(city || "").trim(), String(state || "").trim(), "USA"].filter(Boolean).join(", ");
  if (!query) return null;

  var cacheKey = "qr_geo_" + query.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  try {
    var cache = CacheService.getScriptCache();
    var cached = cache.get(cacheKey);
    if (cached) {
      var parsed = JSON.parse(cached);
      if (parsed && parsed.lat && parsed.lon) return parsed;
    }
  } catch (err) {}

  try {
    var url = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=" + encodeURIComponent(query);
    var resp = UrlFetchApp.fetch(url, {
      method: "get",
      muteHttpExceptions: true,
      headers: {
        "User-Agent": "BRINC QR Web App",
        "Accept-Language": "en-US,en;q=0.9"
      }
    });
    if (resp && resp.getResponseCode && resp.getResponseCode() === 200) {
      var data = JSON.parse(resp.getContentText() || "[]");
      if (data && data.length) {
        var result = {
          lat: String(data[0].lat || ""),
          lon: String(data[0].lon || ""),
          display_name: String(data[0].display_name || query)
        };
        try {
          CacheService.getScriptCache().put(cacheKey, JSON.stringify(result), 21600);
        } catch (cacheErr) {}
        return result;
      }
    }
  } catch (err) {}

  return null;
}

function buildStaticMapUrl_(city, state, coords) {
  var lat = coords && coords.lat ? String(coords.lat) : "";
  var lon = coords && coords.lon ? String(coords.lon) : "";
  if (!lat || !lon) return "";
  var marker = encodeURIComponent(lat + "," + lon + ",red-pushpin");
  var center = encodeURIComponent(lat + "," + lon);
  return "https://staticmap.openstreetmap.de/staticmap.php?center=" + center +
    "&zoom=11&size=980x520&markers=" + marker;
}

function buildLandingPage_(data) {
  var city = escapeHtml_(data.city);
  var state = escapeHtml_(data.state);
  var department = escapeHtml_(data.department);
  var repName = escapeHtml_(data.repName || "BRINC Representative");
  var repEmail = String(data.repEmail || "").trim();
  var repPhone = String(data.repPhone || "").trim();
  var repPhoneDigits = repPhone.replace(/[^\d+]/g, "");
  var repEmailHtml = repEmail ? '<a href="mailto:' + escapeHtml_(repEmail) + '">' + escapeHtml_(repEmail) + '</a>' : "&mdash;";
  var location = [city, state].filter(Boolean).join(", ") || "Coverage overview";
  var reportId = escapeHtml_(data.reportId || data.publicReport || "—");
  var publicReport = escapeHtml_(data.publicReport || "");
  var device = escapeHtml_(data.device || "—");
  var timestamp = escapeHtml_(data.timestamp || "—");
  var sourceApp = escapeHtml_(data.sourceApp || "BRINC QR Web App");
  var sourceUrl = String(data.sourceUrl || "").trim();
  var mailtoSubject = encodeURIComponent("DFR summary follow-up - " + location);
  var mailtoBody = encodeURIComponent(
    "Hi " + (data.repName || "BRINC Representative") + ",\n\n" +
    "I reviewed the DFR summary for " + location + " and would like to continue the conversation.\n\n" +
    "Agency:\nBest callback number:\n\nThanks,"
  );
  var emailLink = repEmail ? 'mailto:' + repEmail + '?subject=' + mailtoSubject + '&body=' + mailtoBody : "";
  var phoneLink = repPhoneDigits ? 'tel:' + repPhoneDigits : "";
  var coords = lookupCoordinates_(city, state);
  var mapUrl = buildStaticMapUrl_(city, state, coords);
  var hasMap = !!mapUrl;
  var mapCaption = coords && coords.display_name
    ? escapeHtml_(coords.display_name)
    : (city || state ? "Coverage preview centered on " + location : "Coverage preview");
  var summaryLine = location
    ? "Here is the customer-facing summary for " + location + "."
    : "Here is the customer-facing summary.";

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
    '    .shell { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 18px; }',
    '    .card { width: min(1040px, 100%); background: var(--panel); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); overflow: hidden; position: relative; }',
    '    .card:before { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,210,255,0.08), transparent 35%); pointer-events: none; }',
    '    .hero { padding: 28px 26px 22px; border-bottom: 1px solid rgba(255,255,255,0.06); }',
    '    .eyebrow-row { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }',
    '    .eyebrow { color: var(--accent); font-size: 12px; letter-spacing: 0.24em; text-transform: uppercase; font-weight: 800; margin-bottom: 10px; }',
    '    h1 { margin: 0; font-size: clamp(32px, 5vw, 56px); line-height: 1.04; }',
    '    .subtitle { margin-top: 10px; color: var(--accent-soft); font-weight: 700; letter-spacing: 0.04em; font-size: 15px; line-height: 1.45; }',
    '    .location { margin-top: 8px; color: var(--muted); font-size: 15px; }',
    '    .body { padding: 18px; display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }',
    '    .panel { background: var(--panel-soft); border: 1px solid rgba(255,255,255,0.05); border-radius: 22px; padding: 18px; }',
    '    .panel.flush { padding: 0; overflow: hidden; }',
    '    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }',
    '    .meta { padding: 14px 16px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); }',
    '    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }',
    '    .value { font-size: 15px; line-height: 1.45; word-break: break-word; }',
    '    .body-copy { color: var(--muted); font-size: 15px; line-height: 1.65; }',
    '    .section-title { color: var(--accent-soft); font-size: 12px; letter-spacing: 0.22em; text-transform: uppercase; font-weight: 800; margin-bottom: 12px; }',
    '    .map-shell { position: relative; background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)); border-radius: 22px; overflow: hidden; min-height: 280px; }',
    '    .map-shell img { width: 100%; height: 100%; display: block; object-fit: cover; }',
    '    .map-fallback { min-height: 180px; display:flex; align-items:center; justify-content:center; padding: 24px; text-align:center; color: var(--muted); }',
    '    .map-overlay { position:absolute; left:0; right:0; bottom:0; padding:14px 16px; background: linear-gradient(180deg, transparent, rgba(5,11,20,0.92) 38%); }',
    '    .map-overlay strong { display:block; color: var(--text); font-size: 16px; margin-bottom: 4px; }',
    '    .map-overlay span { color: var(--muted); font-size: 13px; line-height: 1.4; }',
    '    .cta-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }',
    '    .btn { display: inline-flex; align-items: center; justify-content: center; min-height: 50px; padding: 0 16px; border-radius: 16px; text-decoration: none; font-weight: 800; letter-spacing: 0.01em; flex: 1 1 0; min-width: 0; text-align: center; }',
    '    .btn-primary { background: #f7fbff; color: #07101c; }',
    '    .btn-secondary { border: 1px solid rgba(255,255,255,0.14); color: var(--text); background: transparent; }',
    '    .cta-stack { display:grid; gap:10px; }',
    '    .tone { margin-top: 14px; color: var(--muted); font-size: 14px; line-height: 1.65; }',
    '    .fineprint { margin-top: 14px; color: #6f8094; font-size: 12px; line-height: 1.5; }',
    '    .mini-list { display:grid; gap:10px; margin-top: 14px; }',
    '    .mini-row { padding: 13px 14px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); }',
    '    .mini-row .k { color: var(--muted); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 5px; }',
    '    .mini-row .v { color: var(--text); font-size: 15px; line-height: 1.45; word-break: break-word; }',
    '    .contact-card { display:grid; gap:12px; }',
    '    .contact-name { font-size: clamp(26px, 6vw, 38px); line-height: 1.05; font-weight: 900; }',
    '    .contact-note { color: var(--muted); font-size: 15px; line-height: 1.65; }',
    '    @media (max-width: 720px) {',
    '      .shell { padding: 12px; }',
    '      .hero, .body { padding-left: 14px; padding-right: 14px; }',
    '      .body { grid-template-columns: 1fr; }',
    '      .grid { grid-template-columns: 1fr; }',
    '      .btn { width: 100%; flex-basis: 100%; }',
    '    }',
    '  </style>',
    '</head>',
    '<body>',
    '  <main class="shell">',
    '    <section class="card">',
      '      <header class="hero">',
    '        <div class="eyebrow-row">',
    '          <div class="eyebrow">Drone as a First Responder</div>',
    '          <div class="location">Customer-facing summary</div>',
    '        </div>',
    '        <h1>' + (city || "BRINC DFR") + '</h1>',
    '        <div class="subtitle">' + escapeHtml_(summaryLine) + '</div>',
    '        <div class="location">' + location + '</div>',
    '      </header>',
    '      <div class="body">',
    (hasMap
      ? '        <section class="panel flush">' +
        '<div class="map-shell"><img src="' + escapeHtml_(mapUrl) + '" alt="Map preview for ' + escapeHtml_(location) + '"><div class="map-overlay"><strong>Coverage preview</strong><span>' + mapCaption + '</span></div></div>' +
        '</section>'
      : ''),
    '        <section class="panel">',
    '          <div class="section-title">Base information</div>',
    '          <div class="grid">',
    '            <div class="meta"><div class="label">Representative</div><div class="value">' + repName + '</div></div>',
    '            <div class="meta"><div class="label">Department</div><div class="value">' + (department || "&mdash;") + '</div></div>',
    '            <div class="meta"><div class="label">Email</div><div class="value">' + repEmailHtml + '</div></div>',
    '            <div class="meta"><div class="label">Report ID</div><div class="value">' + (reportId || publicReport || "&mdash;") + '</div></div>',
    '            <div class="meta"><div class="label">Device</div><div class="value">' + (device || "&mdash;") + '</div></div>',
    '            <div class="meta"><div class="label">Scan Time</div><div class="value">' + (timestamp || "&mdash;") + '</div></div>',
    '          </div>',
    '          <div class="tone">This summary is designed to give the customer a fast, mobile-friendly overview without overwhelming them with internal detail.</div>',
    '        </section>',
        '        <section class="panel contact-card">',
    '          <div class="section-title">Next step</div>',
    '          <div class="contact-name">' + repName + '</div>',
    '          <div class="contact-note">Tap a button to continue the conversation with the account executive. The primary action opens a prefilled email so the customer can request a walkthrough or reply with follow-up details.</div>',
    '          <div class="cta-row">',
    (emailLink ? '            <a class="btn btn-primary" href="' + escapeHtml_(emailLink) + '">Email account executive</a>' : ''),
    (phoneLink ? '            <a class="btn btn-secondary" href="' + escapeHtml_(phoneLink) + '">Call the rep</a>' : '            <a class="btn btn-secondary" href="mailto:sales@brincdrones.com?subject=DFR%20follow-up%20-%20' + encodeURIComponent(location) + '">Contact BRINC</a>'),
    '          </div>',
    '          <div class="mini-list">',
    '            <div class="mini-row"><div class="k">Location</div><div class="v">' + location + '</div></div>',
    '            <div class="mini-row"><div class="k">Prepared by</div><div class="v">' + sourceApp + '</div></div>',
    '          </div>',
    '          <div class="fineprint">Prepared at ' + timestamp + '. ' + (sourceUrl ? 'Source: ' + escapeHtml_(sourceUrl) : '') + '</div>',
    '        </section>',
    '      </div>',
    '    </section>',
    '  </main>',
    '</body>',
    '</html>'
  ].join("\n");
}

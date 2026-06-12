import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const ppeCards = [
  { key: "helmet", title: "Helmet", tone: "cyan" },
  { key: "vest", title: "Vest", tone: "magenta" },
  { key: "gloves", title: "Gloves", tone: "gold" },
  { key: "goggles", title: "Goggles", tone: "orange" },
];

function formatScore(score) {
  return typeof score === "number" ? `${(score * 100).toFixed(0)}%` : "--";
}

function formatFrameProgress(current, required) {
  return `${Math.min(current, required)}/${required}`;
}

function StatusCard({ title, status, tone }) {
  const detected = Boolean(status?.detected);
  return (
    <article className={`status-card tone-${tone} ${detected ? "is-on" : "is-off"}`}>
      <div className="status-card__check">{detected ? "✓" : "•"}</div>
      <div className="status-card__body">
        <p className="status-card__title">{title}</p>
        <p className="status-card__state">{detected ? "Detected" : "Missing"}</p>
      </div>
      <div className="status-card__score">{formatScore(status?.score)}</div>
    </article>
  );
}

function AccessSwitch({ authorization, compact = false }) {
  const requiredFrames = authorization?.required_frames ?? 3;
  const consecutiveFrames = authorization?.consecutive_frames ?? 0;
  const switchOn = Boolean(authorization?.switch_on);
  const knownPerson = Boolean(authorization?.known_person);
  const ppeComplete = Boolean(authorization?.ppe_complete);

  let caption = "Waiting for a compliant, known worker.";
  if (switchOn) {
    caption = `${authorization?.authorized_name || "Authorized worker"} validated. Door is open.`;
  } else if (knownPerson && !ppeComplete) {
    caption = "Known person detected, but full PPE is not complete yet.";
  } else if (!knownPerson && ppeComplete) {
    caption = "PPE is complete, but the person is not recognized yet.";
  } else if (knownPerson && ppeComplete) {
    caption = "Authorization conditions are met. Holding steady for 3 consecutive frames.";
  }

  return (
    <section className={`access-switch ${switchOn ? "is-on" : "is-off"} ${compact ? "is-compact" : ""}`}>
      <div className="access-switch__header">
        <div>
          <p className="panel__eyebrow">Entry Control</p>
          <h3>{switchOn ? "Door Unlocked" : "Door Locked"}</h3>
        </div>
        <div className={`switch-visual ${switchOn ? "is-on" : "is-off"}`} aria-hidden="true">
          <div className="switch-visual__track">
            <div className="switch-visual__thumb" />
          </div>
        </div>
      </div>

      <p className="access-switch__caption">{caption}</p>

      <div className="access-switch__stats">
        <div className="access-chip">
          <span>Known Person</span>
          <strong>{knownPerson ? "Yes" : "No"}</strong>
        </div>
        <div className="access-chip">
          <span>Full PPE</span>
          <strong>{ppeComplete ? "Yes" : "No"}</strong>
        </div>
        <div className="access-chip">
          <span>Frame Streak</span>
          <strong>{formatFrameProgress(consecutiveFrames, requiredFrames)}</strong>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [status, setStatus] = useState({
    ready: false,
    device: "loading",
    person_count: 0,
    primary_person: null,
    models: {
      helmet: false,
      vest: false,
      accessory: false,
    },
    error: null,
    authorization: {
      required_frames: 3,
      consecutive_frames: 0,
      eligible: false,
      known_person: false,
      ppe_complete: false,
      switch_on: false,
      door_open: false,
      authorized_name: null,
      last_opened_at: null,
    },
  });
  const [frameToken, setFrameToken] = useState(Date.now());
  const [logoToken, setLogoToken] = useState(Date.now());

  useEffect(() => {
    let mounted = true;

    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/status`, { cache: "no-store" });
        const data = await response.json();
        if (mounted) {
          setStatus(data);
        }
      } catch (_error) {
        if (mounted) {
          setStatus((current) => ({
            ...current,
            ready: false,
          }));
        }
      }
    };

    fetchStatus();
    const timer = window.setInterval(fetchStatus, 750);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setFrameToken(Date.now());
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  const primaryPpe = status.primary_person?.ppe ?? {};
  const authorization = status.authorization ?? {};
  const complianceCount = useMemo(
    () => ppeCards.filter((item) => primaryPpe[item.key]?.detected).length,
    [primaryPpe],
  );

  return (
    <main className="page-shell">
      <div className="backdrop backdrop-one" />
      <div className="backdrop backdrop-two" />

      <section className="hero-panel">
        <div className="hero-copy">
          <div className="brand-row">
            <img
              className="brand-row__logo"
              src={`${API_BASE_URL}/api/branding/logo?t=${logoToken}`}
              alt="SCAI Systems logo"
              onError={() => {
                window.setTimeout(() => {
                  setLogoToken(Date.now());
                }, 500);
              }}
            />
            <div className="brand-row__copy">
              <span className="eyebrow">SCAI Systems</span>
              <p className="brand-row__tag">Industrial Vision Control</p>
            </div>
          </div>
          <h1>Worker Protection Monitoring Dashboard</h1>
          <p className="hero-text">
            Live camera stream with real-time protection checks for helmet, vest, gloves, and goggles.
          </p>

          <AccessSwitch authorization={authorization} compact />

          <div className="hero-metrics">
            <div className="metric-pill">
              <span className="metric-pill__label">Device</span>
              <strong>{status.device}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">People In Frame</span>
              <strong>{status.person_count}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">PPE Ready</span>
              <strong>{`${complianceCount}/4`}</strong>
            </div>
          </div>
        </div>

        <div className="stream-card">
          <div className="stream-card__header">
            <div>
              <p className="stream-card__eyebrow">Live Stream</p>
              <h2>Camera Feed</h2>
            </div>
            <span className={`live-badge ${status.ready ? "is-live" : "is-waiting"}`}>
              <span className="live-dot" />
              {status.ready ? "Live" : "Waiting"}
            </span>
          </div>

          <div className="stream-frame">
            <img src={`${API_BASE_URL}/api/frame?t=${frameToken}`} alt="Live PPE camera feed" />
            {!status.ready && (
              <div className="stream-frame__overlay">
                <p>{status.error ? "Camera stream error" : "Waiting for camera frames..."}</p>
                {status.error && <span>{status.error}</span>}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel panel--merged">
          <div className="panel__header">
            <p className="panel__eyebrow">Primary Worker</p>
            <h3>{status.primary_person?.name || "Protection Checklist"}</h3>
          </div>

          <div className="stack-list stack-list--inline">
            <div className="stack-row">
              <span>Main PPE Logic</span>
              <strong>{status.ready ? "Active" : "Starting"}</strong>
            </div>
            <div className="stack-row">
              <span>Helmet Model</span>
              <strong>{status.models?.helmet ? "Loaded" : "Missing"}</strong>
            </div>
            <div className="stack-row">
              <span>Vest Model</span>
              <strong>{status.models?.vest ? "Loaded" : "Missing"}</strong>
            </div>
            <div className="stack-row">
              <span>Gloves + Goggles</span>
              <strong>{status.models?.accessory ? "Loaded" : "Missing"}</strong>
            </div>
            <div className="stack-row">
              <span>Access Switch</span>
              <strong>{authorization.switch_on ? "On" : "Off"}</strong>
            </div>
          </div>

          <div className="status-grid">
            {ppeCards.map((card) => (
              <StatusCard
                key={card.key}
                title={card.title}
                status={primaryPpe[card.key]}
                tone={card.tone}
              />
            ))}
          </div>

          <div className="summary-card">
            <p className="summary-card__eyebrow">Current Snapshot</p>
            <h4>
              {authorization.switch_on
                ? `Door opened for ${authorization.authorized_name}`
                : status.primary_person
                  ? status.primary_person.name
                  : "No worker detected"}
            </h4>
            <p>
              {authorization.switch_on
                ? `Authorization completed after ${authorization.required_frames} compliant frames.`
                : status.primary_person
                ? `${status.primary_person.name_confidence ? `Face match ${formatScore(status.primary_person.name_confidence)}.` : ""} Detection confidence ${formatScore(status.primary_person.confidence)}`
                : status.error || "Stand in front of the camera to populate the protection cards."}
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

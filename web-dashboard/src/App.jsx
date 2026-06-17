import { useEffect, useMemo, useState } from "react";
import cameraPlaceholder from "./camera-placeholder.svg";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const ppeCards = [
  { key: "helmet", title: "Casque", tone: "cyan" },
  { key: "vest", title: "Gilet", tone: "magenta" },
  { key: "gloves", title: "Gants", tone: "gold" },
  { key: "goggles", title: "Lunettes", tone: "orange" },
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
      <div className="status-card__check">{detected ? "OK" : "-"}</div>
      <div className="status-card__body">
        <p className="status-card__title">{title}</p>
        <p className="status-card__state">{detected ? "Detecte" : "Absent"}</p>
      </div>
      <div className="status-card__score">{formatScore(status?.score)}</div>
    </article>
  );
}

function ThreatCard({ detected, count }) {
  return (
    <article className={`threat-card ${detected ? "is-alert" : "is-clear"}`}>
      <div className="threat-card__badge">{detected ? "!" : "OK"}</div>
      <div className="threat-card__body">
        <p className="threat-card__title">Threat Detected</p>
        <p className="threat-card__state">
          {detected ? "Mobile or screen spoof attempt detected" : "No spoofing threat detected"}
        </p>
      </div>
      <div className="threat-card__count">{count || 0}</div>
    </article>
  );
}

function AccessSwitch({ authorization, compact = false }) {
  const requiredFrames = authorization?.required_frames ?? 2;
  const consecutiveFrames = authorization?.consecutive_frames ?? 0;
  const switchOn = Boolean(authorization?.switch_on);
  const knownPerson = Boolean(authorization?.known_person);
  const ppeComplete = Boolean(authorization?.ppe_complete);

  let caption = "En attente d'un travailleur connu et conforme.";
  if (switchOn) {
    caption = `${authorization?.authorized_name || "Travailleur autorise"} valide. La porte est ouverte.`;
  } else if (knownPerson && !ppeComplete) {
    caption = "Personne connue detectee, mais l'EPI complet n'est pas encore valide.";
  } else if (!knownPerson && ppeComplete) {
    caption = "L'EPI est complet, mais la personne n'est pas encore reconnue.";
  } else if (knownPerson && ppeComplete) {
    caption = "Les conditions d'autorisation sont remplies. Maintien sur 2 images consecutives.";
  }

  return (
    <section className={`access-switch ${switchOn ? "is-on" : "is-off"} ${compact ? "is-compact" : ""}`}>
      <div className="access-switch__header">
        <div>
          <p className="panel__eyebrow">Controle d'acces</p>
          <h3>{switchOn ? "Porte Deverrouillee" : "Porte Verrouillee"}</h3>
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
          <span>Personne Connue</span>
          <strong>{knownPerson ? "Oui" : "Non"}</strong>
        </div>
        <div className="access-chip">
          <span>EPI Complet</span>
          <strong>{ppeComplete ? "Oui" : "Non"}</strong>
        </div>
        <div className="access-chip">
          <span>Serie d'Images</span>
          <strong>{formatFrameProgress(consecutiveFrames, requiredFrames)}</strong>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [status, setStatus] = useState({
    ready: false,
    device: "chargement",
    person_count: 0,
    primary_person: null,
    threat_detected: false,
    threat_count: 0,
    models: {
      helmet: false,
      vest: false,
      accessory: false,
    },
    error: null,
    authorization: {
      required_frames: 2,
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
    const timer = window.setInterval(fetchStatus, 400);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
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
              alt="Logo SCAI Systems"
              onError={() => {
                window.setTimeout(() => {
                  setLogoToken(Date.now());
                }, 500);
              }}
            />
            <div className="brand-row__copy">
              <span className="eyebrow">SCAI Systems</span>
              <p className="brand-row__tag">Controle Visuel Industriel</p>
            </div>
          </div>
          <h1>Surveillance de la Protection des collaborateurs</h1>
          <p className="hero-text">
            Flux camera en direct avec verification en temps reel du casque, du gilet, des gants et des lunettes.
          </p>

          <AccessSwitch authorization={authorization} compact />

          <div className="hero-metrics">
            <div className="metric-pill">
              <span className="metric-pill__label">Appareil</span>
              <strong>{status.device}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">Personnes a l'Ecran</span>
              <strong>{status.person_count}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">EPI Valides</span>
              <strong>{`${complianceCount}/4`}</strong>
            </div>
          </div>
        </div>

        <div className="stream-card">
          <div className="stream-card__header">
            <div>
              <p className="stream-card__eyebrow">Flux en Direct</p>
              <h2>Vue Camera</h2>
            </div>
            <span className={`live-badge ${status.ready ? "is-live" : "is-waiting"}`}>
              <span className="live-dot" />
              {status.ready ? "En Direct" : "En Attente"}
            </span>
          </div>

          <div className="stream-frame">
            <img
              src={status.ready ? `${API_BASE_URL}/video_feed` : cameraPlaceholder}
              alt={status.ready ? "Flux camera EPI en direct" : "Illustration d'attente de la camera"}
            />
            {!status.ready && (
              <div className="stream-frame__overlay">
                <p>{status.error ? "Erreur du flux camera" : "En attente des images camera..."}</p>
                {status.error && <span>{status.error}</span>}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel panel--merged">
          <div className="panel__header">
            <p className="panel__eyebrow">Travailleur Principal</p>
            <h3>{status.primary_person?.name || "Checklist de Protection"}</h3>
          </div>

          <ThreatCard detected={status.threat_detected} count={status.threat_count} />

          <div className="stack-list stack-list--inline">
            <div className="stack-row">
              <span>Logique EPI Principale</span>
              <strong>{status.ready ? "Active" : "Demarrage"}</strong>
            </div>
            <div className="stack-row">
              <span>Modele Casque</span>
              <strong>{status.models?.helmet ? "Charge" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Modele Gilet</span>
              <strong>{status.models?.vest ? "Charge" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Gants + Lunettes</span>
              <strong>{status.models?.accessory ? "Charge" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Interrupteur d'Acces</span>
              <strong>{authorization.switch_on ? "Active" : "Desactive"}</strong>
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
            <p className="summary-card__eyebrow">Etat Actuel</p>
            <h4>
              {status.threat_detected
                ? "Threat detected on mobile or screen"
                : authorization.switch_on
                ? `Porte ouverte pour ${authorization.authorized_name}`
                : status.primary_person
                  ? status.primary_person.name
                  : "Aucun travailleur detecte"}
            </h4>
            <p>
              {status.threat_detected
                ? `${status.threat_count || 0} spoof attempt(s) detected. Access remains blocked until a real face is seen.`
                : authorization.switch_on
                ? `Autorisation validee apres ${authorization.required_frames} images conformes.`
                : status.primary_person
                  ? `${status.primary_person.name_confidence ? `Correspondance visage ${formatScore(status.primary_person.name_confidence)}.` : ""} Confiance de detection ${formatScore(status.primary_person.confidence)}`
                  : status.error || "Placez-vous devant la camera pour remplir les cartes de protection."}
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

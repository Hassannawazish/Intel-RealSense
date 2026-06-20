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
        <p className="status-card__state">{detected ? "Détecté" : "Absent"}</p>
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
        <p className="threat-card__title">Menace détectée</p>
        <p className="threat-card__state">
          {detected ? "Tentative d'usurpation détectée sur mobile ou écran" : "Aucune tentative d'usurpation détectée"}
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
    caption = `${authorization?.authorized_name || "Collaborateur autorisé"} validé. La porte est ouverte.`;
  } else if (knownPerson && !ppeComplete) {
    caption = "Personne connue détectée, mais l'ÉPI complet n'est pas encore validé.";
  } else if (!knownPerson && ppeComplete) {
    caption = "L'ÉPI est complet, mais la personne n'est pas encore reconnue.";
  } else if (knownPerson && ppeComplete) {
    caption = "Les conditions d'autorisation sont remplies. Maintien sur 2 images consécutives.";
  }

  return (
    <section className={`access-switch ${switchOn ? "is-on" : "is-off"} ${compact ? "is-compact" : ""}`}>
      <div className="access-switch__header">
        <div>
          <p className="panel__eyebrow">Contrôle d'accès</p>
          <h3>{switchOn ? "Porte déverrouillée" : "Porte verrouillée"}</h3>
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
          <span>ÉPI complet</span>
          <strong>{ppeComplete ? "Oui" : "Non"}</strong>
        </div>
        <div className="access-chip">
          <span>Série d'images</span>
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
              <p className="brand-row__tag">Contrôle visuel industriel</p>
            </div>
          </div>
          <h1>Surveillance de la Protection des collaborateurs</h1>
          <p className="hero-text">
            Flux caméra en direct avec vérification en temps réel du casque, du gilet, des gants et des lunettes.
          </p>

          <AccessSwitch authorization={authorization} compact />

          <div className="hero-metrics">
            <div className="metric-pill">
              <span className="metric-pill__label">Appareil</span>
              <strong>{status.device}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">Personnes à l'écran</span>
              <strong>{status.person_count}</strong>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">ÉPI validés</span>
              <strong>{`${complianceCount}/4`}</strong>
            </div>
          </div>
        </div>

        <div className="stream-card">
          <div className="stream-card__header">
            <div>
              <p className="stream-card__eyebrow">Flux en direct</p>
              <h2>Vue caméra</h2>
            </div>
            <span className={`live-badge ${status.ready ? "is-live" : "is-waiting"}`}>
              <span className="live-dot" />
              {status.ready ? "En direct" : "En attente"}
            </span>
          </div>

          <div className="stream-frame">
            <img
              src={status.ready ? `${API_BASE_URL}/video_feed` : cameraPlaceholder}
              alt={status.ready ? "Flux caméra ÉPI en direct" : "Illustration d'attente de la caméra"}
            />
            {!status.ready && (
              <div className="stream-frame__overlay">
                <p>{status.error ? "Erreur du flux caméra" : "En attente des images caméra..."}</p>
                {status.error && <span>{status.error}</span>}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="panel panel--merged">
          <div className="panel__header">
            <p className="panel__eyebrow">Collaborateur principal</p>
            <h3>{status.primary_person?.name || "Checklist de protection"}</h3>
          </div>

          <ThreatCard detected={status.threat_detected} count={status.threat_count} />

          <div className="stack-list stack-list--inline">
            <div className="stack-row">
              <span>Logique ÉPI principale</span>
              <strong>{status.ready ? "Active" : "Démarrage"}</strong>
            </div>
            <div className="stack-row">
              <span>Modèle casque</span>
              <strong>{status.models?.helmet ? "Chargé" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Modèle gilet</span>
              <strong>{status.models?.vest ? "Chargé" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Gants + Lunettes</span>
              <strong>{status.models?.accessory ? "Chargé" : "Absent"}</strong>
            </div>
            <div className="stack-row">
              <span>Interrupteur d'accès</span>
              <strong>{authorization.switch_on ? "Activé" : "Désactivé"}</strong>
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
            <p className="summary-card__eyebrow">État actuel</p>
            <h4>
              {status.threat_detected
                ? "Menace détectée sur mobile ou écran"
                : authorization.switch_on
                ? `Porte ouverte pour ${authorization.authorized_name}`
                : status.primary_person
                  ? status.primary_person.name
                  : "Aucun collaborateur détecté"}
            </h4>
            <p>
              {status.threat_detected
                ? `${status.threat_count || 0} tentative(s) d'usurpation détectée(s). L'accès reste bloqué jusqu'à la détection d'un vrai visage.`
                : authorization.switch_on
                ? `Autorisation validée après ${authorization.required_frames} images conformes.`
                : status.primary_person
                  ? `${status.primary_person.name_confidence ? `Correspondance du visage ${formatScore(status.primary_person.name_confidence)}.` : ""} Confiance de détection ${formatScore(status.primary_person.confidence)}`
                  : status.error || "Placez-vous devant la caméra pour renseigner les cartes de protection."}
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

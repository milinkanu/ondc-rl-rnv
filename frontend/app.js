/* Configuration */
const API_URL = window.location.origin; // Assume served from FastAPI directly
let sessionId = null;
let currentPhase = 0; // 0: SEARCH, 1: SELECT, 2: INIT, 3: CONFIRM, 4: TRACK, 5: POST_ORDER
let totalReward = 0;

const ACTIONS = [
  { id: 0, name: 'SEARCH_PRODUCTS', label: 'Search Products', icon: 'fa-magnifying-glass', validPhases: [0] },
  { id: 1, name: 'SELECT_SELLER_0', label: 'Select Seller 1', icon: 'fa-hand-pointer', validPhases: [1] },
  { id: 2, name: 'SELECT_SELLER_1', label: 'Select Seller 2', icon: 'fa-hand-pointer', validPhases: [1] },
  { id: 3, name: 'SELECT_SELLER_2', label: 'Select Seller 3', icon: 'fa-hand-pointer', validPhases: [1] },
  { id: 4, name: 'SELECT_SELLER_3', label: 'Select Seller 4', icon: 'fa-hand-pointer', validPhases: [1] },
  { id: 5, name: 'SELECT_SELLER_4', label: 'Select Seller 5', icon: 'fa-hand-pointer', validPhases: [1] },
  { id: 6, name: 'INIT_ORDER', label: 'Init Order', icon: 'fa-cart-arrow-down', validPhases: [2] },
  { id: 7, name: 'CONFIRM_ORDER', label: 'Confirm Order', icon: 'fa-check', validPhases: [3] },
  { id: 8, name: 'CANCEL_BEFORE_CONFIRM', label: 'Cancel (Before Confirm)', icon: 'fa-xmark', validPhases: [3] },
  { id: 9, name: 'TRACK_ORDER', label: 'Track Delivery', icon: 'fa-location-dot', validPhases: [4] },
  { id: 10, name: 'ACCEPT_DELIVERY', label: 'Accept Delivery', icon: 'fa-box-open', validPhases: [5] },
  { id: 11, name: 'CANCEL_ORDER', label: 'Cancel Order', icon: 'fa-ban', validPhases: [4, 5] },
  { id: 12, name: 'RETURN_ITEM', label: 'Return Item', icon: 'fa-rotate-left', validPhases: [5] },
  { id: 13, name: 'FILE_GRIEVANCE', label: 'File Grievance', icon: 'fa-triangle-exclamation', validPhases: [5] },
  { id: 14, name: 'WAIT', label: 'Wait (Tick Market)', icon: 'fa-hourglass-half', validPhases: [0, 1, 2, 3, 4, 5] }
];

/* Elements */
const btnStart = document.getElementById('btn-start');
const statBudget = document.querySelector('#stat-budget .value');
const statUrgency = document.querySelector('#stat-urgency .value');
const statReward = document.querySelector('#stat-reward .value');
const targetItemBadge = document.getElementById('target-item-badge');
const actionContainer = document.getElementById('actions-container');
const sellerContainer = document.getElementById('sellers-container');
const eventLogs = document.getElementById('event-logs');

/* Initialization */
btnStart.addEventListener('click', startSession);

async function startSession() {
    try {
        btnStart.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting...';
        btnStart.disabled = true;

        const res = await fetch(`${API_URL}/session/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                max_steps: 50,
                n_sellers: 5,
                initial_budget: 1000.0,
                target_item: "High-End Laptop",
                urgency: Math.random() // Random urgency
            })
        });

        if (!res.ok) throw new Error('Failed to start session');
        
        const data = await res.json();
        sessionId = data.session_id;
        
        totalReward = 0;
        statReward.innerText = totalReward.toFixed(2);
        statBudget.innerText = '₹' + data.info.budget.toFixed(2);
        statUrgency.innerText = (data.info.urgency * 100).toFixed(0) + '%';
        targetItemBadge.innerText = "Target: High-End Laptop";
        
        eventLogs.innerHTML = ''; // clear logs
        logEvent('info', 'Session started! Explore the market.');
        showToast('success', 'Session Started', 'Ready to navigate ONDC!');

        await fetchState();
    } catch (err) {
        showToast('error', 'Error', err.message);
    } finally {
        btnStart.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Restart Session';
        btnStart.disabled = false;
    }
}

async function fetchState() {
    if (!sessionId) return;
    try {
        const res = await fetch(`${API_URL}/session/${sessionId}/state`);
        if (!res.ok) throw new Error('Failed to fetch state');
        const state = await res.json();
        
        // Update phase
        currentPhase = state.current_phase;
        updatePhaseUI();
        
        // Update Budget
        statBudget.innerText = '₹' + state.budget.toFixed(2);

        // Update Sellers
        renderSellers(state.sellers, state.selected_seller_id);
        
        // Update Action Buttons
        renderActions();
        
    } catch (err) {
        console.error(err);
    }
}

async function sendAction(actionId, actionName) {
    if (!sessionId) return;
    try {
        logEvent('info', `Action Taken: ${actionName}`);
        
        const res = await fetch(`${API_URL}/session/${sessionId}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionId })
        });
        
        if (!res.ok) {
            const errBody = await res.json();
            throw new Error(errBody.detail? errBody.detail.detail || errBody.detail : 'Action failed');
        }
        
        const data = await res.json();
        
        // Parse reward
        if(data.reward !== 0) {
            totalReward += data.reward;
            statReward.innerText = totalReward.toFixed(2);
            const type = data.reward > 0 ? 'reward' : 'penalty';
            const breakdownItems = Object.entries(data.info.reward_breakdown).filter(([k,v]) => v !== 0);
            const reason = breakdownItems.map(([k,v]) => `${k} (${v > 0 ? '+'+v.toFixed(2) : v.toFixed(2)})`).join(', ');
            logEvent(type, `Reward: ${data.reward.toFixed(2)} | ${reason}`);
        }
        
        if(data.info.invalid_action) {
            logEvent('penalty', 'Invalid Action in current phase!');
            showToast('error', 'Invalid Action', 'That action is not allowed right now.');
        }

        if(data.done) {
            logEvent('info', 'Episode Completed.');
            showToast('info', 'Episode Done', `Final Reward: ${totalReward.toFixed(2)}`);
            actionContainer.innerHTML = '<p style="color:var(--text-muted)">Episode Finished! Start a new session.</p>';
        }

        await fetchState();

    } catch(err) {
        showToast('error', 'Action Error', err.message);
        logEvent('penalty', `Error: ${err.message}`);
    }
}

function updatePhaseUI() {
    document.querySelectorAll('.phase-step').forEach(el => {
        const p = parseInt(el.dataset.phase);
        if (p === currentPhase) {
            el.className = 'phase-step active';
        } else if (p < currentPhase) {
            el.className = 'phase-step completed';
        } else {
            el.className = 'phase-step';
        }
    });
}

function renderActions() {
    actionContainer.innerHTML = '';
    
    // Filter valid actions for this phase
    const validActions = ACTIONS.filter(a => a.validPhases.includes(currentPhase));
    
    validActions.forEach(a => {
        // Skip Select Sellers explicitly since we click on the seller cards
        if (a.name.startsWith('SELECT_SELLER')) return;

        const btn = document.createElement('button');
        btn.className = 'btn-action';
        btn.innerHTML = `<i class="fa-solid ${a.icon}"></i> ${a.label}`;
        btn.onclick = () => sendAction(a.id, a.name);
        actionContainer.appendChild(btn);
    });
}

function renderSellers(sellers, selectedId) {
    if(!sellers || sellers.length === 0) {
        sellerContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-shop"></i>
                <p>No sellers visible yet. Hit "Search" or "Wait".</p>
            </div>`;
        return;
    }
    
    sellerContainer.innerHTML = '';
    let renderedCount = 0;
    
    sellers.forEach((s, idx) => {
        // Hide missing sellers unless selected
        if (!s.is_available && currentPhase < 1) return;
        renderedCount++;

        const isSelected = selectedId === idx;
        const available = s.is_available;
        
        const card = document.createElement('div');
        card.className = `seller-card ${isSelected ? 'seller-target' : ''}`;
        
        // Click to select
        if (currentPhase === 1 && available && !isSelected) {
            card.onclick = () => sendAction(idx + 1, `SELECT_SELLER_${idx}`);
            card.title = "Click to select this seller";
        }
        
        card.innerHTML = `
            <div class="card-header">
                <div class="card-title">Seller Block ${idx + 1}</div>
                <div class="card-rating">
                    <i class="fa-solid fa-star"></i> ${s.rating.toFixed(1)}
                </div>
            </div>
            <div class="card-details">
                <div class="detail-row price">
                    <span class="label">Offer Price</span>
                    <span class="val">₹${s.price.toFixed(2)}</span>
                </div>
                <div class="detail-row eta">
                    <span class="label">Est. Time</span>
                    <span class="val">${s.delivery_eta} steps</span>
                </div>
                <div class="detail-row stock">
                    <span class="label">Stock Available</span>
                    <span class="val">${s.stock} pcs</span>
                </div>
                <div style="margin-top:0.5rem; text-align:right">
                    ${available ? 
                        '<span class="badge badge-available">Online</span>' : 
                        '<span class="badge badge-unavailable">Offline/Out of Stock</span>'}
                    ${isSelected ? '<span class="badge badge-available" style="margin-left:0.5rem"><i class="fa-solid fa-check"></i> Selected</span>' : ''}
                </div>
            </div>
        `;
        sellerContainer.appendChild(card);
    });

    if(renderedCount === 0) {
        sellerContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-ghost"></i>
                <p>All sellers are currently offline or out of stock.</p>
            </div>`;
    }
}

function logEvent(type, text) {
    const time = new Date().toLocaleTimeString([], {hour12: false});
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    let content = text;
    if(type === 'reward' || type === 'penalty') {
        const match = text.match(/Reward: ([-\d.]+)\s*\|\s*(.*)/);
        if(match) {
            content = `<span class="reward-val">${match[1]}</span> <span class="message">${match[2]}</span>`;
        }
    }
    
    entry.innerHTML = `
        <div class="log-header">
            <span class="time">${time}</span>
        </div>
        <div class="message">${content}</div>
    `;
    eventLogs.prepend(entry);
}

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'fa-info-circle';
    if(type === 'success') icon = 'fa-check-circle';
    if(type === 'error') icon = 'fa-xmark-circle';
    if(type === 'warning') icon = 'fa-exclamation-triangle';
    
    toast.innerHTML = `
        <i class="fa-solid ${icon} fa-2x"></i>
        <div class="toast-content">
            <h4>${title}</h4>
            <p>${message}</p>
        </div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastFade 0.3s forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

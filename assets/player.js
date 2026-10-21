/* One captioned playlist player shared by the course hub and lecture pages. */
(() => {
  const mediaRoot = new URL('../videos/', document.currentScript.src);
  const dialog = document.createElement('dialog');
  dialog.className = 'course-player';
  dialog.setAttribute('aria-labelledby', 'player-title');
  dialog.innerHTML = `<video controls playsinline preload="none" aria-label="Lesson video"></video>
    <div class="player-bar"><b id="player-title"></b><button class="cbtn">Close</button></div>
    <p class="player-error" role="status" hidden></p><div class="player-list" aria-label="Video parts"></div>`;
  document.body.append(dialog);
  const video = dialog.querySelector('video');
  const list = dialog.querySelector('.player-list');
  const error = dialog.querySelector('.player-error');
  let course, current = 0, opener;

  function playPart(index) {
    current = index;
    const part = course.parts[index];
    error.hidden = true;
    video.pause();
    video.replaceChildren();
    const src = new URL(part.src.split('/').pop(), mediaRoot).href;
    for (const [lang, label] of [['en', 'English'], ['zh', '中文']]) {
      const track = document.createElement('track');
      Object.assign(track, {kind: 'subtitles', srclang: lang, label,
        src: src.replace(/\.mp4$/, `.${lang}.vtt`), default: lang === 'en'});
      video.append(track);
    }
    video.src = src;
    dialog.querySelector('b').textContent = `${course.title} · Part ${index + 1} of ${course.parts.length} — ${part.title}`;
    [...list.children].forEach((button, i) => {
      button.classList.toggle('on', i === index);
      button.setAttribute('aria-pressed', String(i === index));
    });
    video.play().catch(() => { /* Native controls remain available if autoplay is blocked. */ });
  }

  video.addEventListener('ended', () => {
    if (current + 1 < course.parts.length) playPart(current + 1);
  });
  video.addEventListener('error', () => {
    error.textContent = 'This video could not load. Try another part or reopen the player.';
    error.hidden = false;
  });
  dialog.querySelector('.player-bar button').onclick = () => dialog.close();
  dialog.addEventListener('close', () => {
    video.pause();
    video.removeAttribute('src');
    video.replaceChildren();
    video.load();
    if (opener?.isConnected) opener.focus();
  });
  window.addEventListener('hashchange', () => { if (dialog.open) dialog.close(); });
  document.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-course]');
    if (!button) return;
    course = window.DDA_VIDEOS[button.dataset.course];
    if (!course) return;
    opener = button;
    list.replaceChildren(...course.parts.map((part, index) => {
      const item = document.createElement('button');
      item.textContent = `${index + 1}. ${part.title} · ${part.duration}`;
      item.onclick = () => playPart(index);
      return item;
    }));
    dialog.showModal();
    playPart(Number(button.dataset.part || 0));
  });
})();

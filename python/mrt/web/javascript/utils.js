export const roomName = JSON.parse(document.getElementById('room-name').textContent);

export function update_yaml_configurations(e) {
  const data = JSON.parse(e.data);
  for (const [stage, stage_data] of Object.entries(data)) {
    for (const [attr, value] of Object.entries(stage_data)) {
      const id = '#' + stage + '_' + attr;
      document.querySelector(id).value = value;
    }
  }
}

export const yamlUpdateSocket = new WebSocket(
  'ws://'
  + window.location.host
  + '/ws/web/yaml/update/'
  + roomName
  + '/'
);

yamlUpdateSocket.addEventListener(
  'message', update_yaml_configurations);

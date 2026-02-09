## Memory Reading & Game State Extraction

### Citra Memory Structure

Pokemon Sun/Moon memory addresses (these may vary by ROM version):

```python
class PokemonMemoryReader:
    """Read Pokemon Sun/Moon game state from Citra memory"""
    
    def __init__(self, process_name="citra-qt.exe"):
        self.pm = pymem.Pymem(process_name)
        self.base_address = self.find_base_address()
        
        # Memory offsets (EXAMPLE - you'll need to find actual values)
        self.offsets = {
            'player_x': 0x330D9838,
            'player_y': 0x330D983C,
            'player_z': 0x330D9840,
            'map_id': 0x330D9844,
            'money': 0x33015AB0,
            'party_count': 0x33015A3C,
            'party_data': 0x33015A40,  # Start of party Pokemon data
            'pokedex_owned': 0x33016000,
            'pokedex_seen': 0x33016400,
            'story_flags': 0x33017000,
            'battle_state': 0x3301A000,
        }
    
    def find_base_address(self):
        """Find base address of game memory"""
        # This requires reverse engineering the game
        # Tools: Cheat Engine, Citra's debugger
        pass
    
    def read_player_position(self):
        """Get player's current position"""
        x = self.read_float(self.offsets['player_x'])
        y = self.read_float(self.offsets['player_y'])
        z = self.read_float(self.offsets['player_z'])
        map_id = self.read_int(self.offsets['map_id'])
        return (x, y, z, map_id)
    
    def read_party_data(self):
        """Read all party Pokemon data"""
        party_count = self.read_int(self.offsets['party_count'])
        party = []
        
        for i in range(min(party_count, 6)):
            offset = self.offsets['party_data'] + (i * 260)  # 260 bytes per Pokemon
            pokemon = {
                'species': self.read_short(offset + 0x08),
                'level': self.read_byte(offset + 0xEC),
                'hp_current': self.read_short(offset + 0xF0),
                'hp_max': self.read_short(offset + 0xF2),
                'attack': self.read_short(offset + 0xF4),
                'defense': self.read_short(offset + 0xF6),
                'status': self.read_int(offset + 0xE8),
                # ... more stats
            }
            party.append(pokemon)
        
        return party
    
    def read_battle_state(self):
        """Check if in battle and get battle info"""
        in_battle = self.read_bool(self.offsets['battle_state'])
        if not in_battle:
            return None
        
        return {
            'in_battle': True,
            'opponent_hp': self.read_short(self.offsets['battle_state'] + 0x10),
            'opponent_hp_max': self.read_short(self.offsets['battle_state'] + 0x12),
            'opponent_level': self.read_byte(self.offsets['battle_state'] + 0x14),
            # ... more battle data
        }
    
    def read_int(self, address):
        return self.pm.read_int(self.base_address + address)
    
    def read_short(self, address):
        return self.pm.read_short(self.base_address + address)
    
    def read_byte(self, address):
        return self.pm.read_uchar(self.base_address + address)
    
    def read_float(self, address):
        return self.pm.read_float(self.base_address + address)
    
    def read_bool(self, address):
        return self.pm.read_bool(self.base_address + address)
```

### Finding Memory Addresses

**Tools:**
1. **Cheat Engine** - Free memory scanner for Windows
2. **Citra's Built-in Debugger** - Can dump memory regions
3. **PokeRadar/PKHeX** - Pokemon-specific save editors (understand data structures)

**Process:**
```
1. Open Pokemon Sun/Moon in Citra
2. Open Cheat Engine, attach to citra-qt.exe process
3. Scan for known values (e.g., current money amount)
4. Change value in game (buy/sell item)
5. Scan for new value
6. Repeat until single address found
7. Document offset from base address
8. Repeat for all needed values
```

### Alternative: Lua Scripting

Citra supports Lua scripting for memory access:

```lua
-- Citra Lua script example
function get_player_position()
    local x = memory.readfloat(0x330D9838)
    local y = memory.readfloat(0x330D983C)
    local z = memory.readfloat(0x330D9840)
    return x, y, z
end

function get_party_count()
    return memory.readbyte(0x33015A3C)
end
```

You can call these scripts from Python using IPC or file-based communication.
